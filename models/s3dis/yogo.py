import time
from collections import OrderedDict
import convpoint.knn.lib.python.nearest_neighbors as knn

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from modules.functional.backend import _backend
from modules.functional.sampling import gather, furthest_point_sample
import modules.functional as mf

__all__ = ['YOGO']

def knn_search(input_pts, query_pts, k):
    knn_idx  = knn.knn_batch(input_pts.permute(0, 2, 1).data.cpu(),
        query_pts.permute(0, 2, 1).data.cpu(), k
        )
    return knn_idx.astype(np.int64)

def conv1x1_1d(inplanes, out_planes, stride=1, groups=1):
    """1x1 convolution"""
    return nn.Conv1d(inplanes, out_planes, kernel_size=1, stride=stride,
                     groups=groups, bias=False)

def conv1x1(inplanes, out_planes, stride=1, groups=1):
    """1x1 convolution"""
    return nn.Conv1d(inplanes, out_planes, kernel_size=1, stride=stride,
                     groups=groups, bias=False)

class MatMul(nn.Module):
    """A wrapper class such that we can count the FLOPs of matmul
    """
    def __init__(self):
        super(MatMul, self).__init__()

    def forward(self, A, B):
        return torch.matmul(A, B)

class Transformer(nn.Module):
    def __init__(self, token_c, t_layer=1, head=2, kqv_groups=1,
                 norm_layer_1d=nn.Identity):
        super(Transformer, self).__init__()

        self.k_conv = nn.ModuleList()
        self.q_conv = nn.ModuleList()
        self.v_conv = nn.ModuleList()
        self.kqv_bn = nn.ModuleList()
        self.kq_matmul = nn.ModuleList()
        self.kqv_matmul = nn.ModuleList()
        self.ff_conv = nn.ModuleList()
        for _ in range(t_layer):
            self.k_conv.append(nn.Sequential(
                conv1x1_1d(token_c, token_c // 2, groups=kqv_groups),
                norm_layer_1d(token_c // 2)
            ))
            self.q_conv.append(nn.Sequential(
                conv1x1_1d(token_c, token_c // 2, groups=kqv_groups),
                norm_layer_1d(token_c // 2)
            ))
            self.v_conv.append(nn.Sequential(
                conv1x1_1d(token_c, token_c, groups=kqv_groups),
                norm_layer_1d(token_c)
            ))
            self.kq_matmul.append(MatMul())
            self.kqv_matmul.append(MatMul())
            self.kqv_bn.append(norm_layer_1d(token_c))
            # zero-init
            #nn.init.constant_(self.kqv_bn[-1].weight, 0)
            self.ff_conv.append(nn.Sequential(
                conv1x1_1d(token_c, token_c * 2),
                norm_layer_1d(token_c * 2),
                nn.ReLU(inplace=True),
                conv1x1_1d(token_c * 2, token_c),
                norm_layer_1d(token_c),
            ))
            # initialize the bn weight to zero to improves the training
            # stability.
            #nn.init.constant_(self.ff_conv[-1][1].weight, 1)

        self.token_c = token_c
        self.t_layer = t_layer
        self.head = head

    def forward(self, x):
        N = x.shape[0]
        for _idx in range(self.t_layer):
            k = self.k_conv[_idx](x).view(
                N, self.head, self.token_c // 2 // self.head, -1)
            q = self.q_conv[_idx](x).view(
                N, self.head, self.token_c // 2 // self.head, -1)
            v = self.v_conv[_idx](x).view(
                N, self.head, self.token_c // self.head, -1)
            # N, h, L, C/h * N, h, C/h, L -> N, h, L, L
            kq = self.kq_matmul[_idx](k.permute(0, 1, 3, 2), q)
            # N, h, L, L
            kq = F.softmax(kq / np.sqrt(self.token_c / 2 / self.head), dim=2)
            # N, h, C/h, L * N, h, L, L -> N, h, C/h, L
            kqv = self.kqv_matmul[_idx](v, kq).view(N, self.token_c, -1)
            kqv = self.kqv_bn[_idx](kqv)
            x = x + kqv
            x = x + self.ff_conv[_idx](x)

        return x

class Projector(nn.Module):
    def __init__(self, token_c, planes, head=2, min_group_planes=64,
                 norm_layer_1d=nn.Identity):
        super(Projector, self).__init__()

        if token_c != planes:
            self.proj_value_conv = nn.Sequential(
                conv1x1_1d(token_c, planes),
                norm_layer_1d(planes))
        else:
            self.proj_value_conv = nn.Identity()

        self.proj_key_conv = nn.Sequential(
            conv1x1_1d(token_c, planes),
            norm_layer_1d(planes)
        )
        self.proj_query_conv = nn.Sequential(
            conv1x1_1d(planes, planes),
            norm_layer_1d(planes)
        )
        self.proj_kq_matmul = MatMul()
        self.proj_matmul = MatMul()
        self.proj_bn = norm_layer_1d(planes)
        # zero-init
        #nn.init.constant_(self.proj_bn.weight, 1)

        self.ff_conv = nn.Sequential(
            conv1x1_1d(planes, 2 * planes),
            norm_layer_1d(2 * planes),
            nn.ReLU(inplace=True),
            conv1x1_1d(2 * planes, planes),
            norm_layer_1d(planes)
            )

        self.head = head

    def forward(self, x, x_t):
        N, _, L = x_t.shape
        h = self.head
        # -> N, h, C/h, L
        proj_v = self.proj_value_conv(x_t).view(N, h, -1, L)
        # -> N, h, C/h, L
        proj_k = self.proj_key_conv(x_t).view(N, h, -1, L)
        proj_q = self.proj_query_conv(x)
        N, C, _ = proj_q.shape
        # -> N, h, HW, c/H
        proj_q = proj_q.view(N, h, C // h, -1).permute(0, 1, 3, 2)
        # N, h, HW, C/h * N, h, C/h, L -> N, h, HW, L
        proj_coef = F.softmax(
            self.proj_kq_matmul(proj_q, proj_k) / np.sqrt(C / h), dim=3)

        # N, h, C/h, L * N, h, L, HW -> N, h, C/h, HW
        x_p = self.proj_matmul(proj_v, proj_coef.permute(0, 1, 3, 2))
        # -> N, C, H, W
        _, _, S = x.shape
        x_p = self.proj_bn(x_p.view(N, -1, S))

        x = x + self.ff_conv(x + x_p)

        return x


class RIM(nn.Module):
    def __init__(self, token_c, input_dims, output_dims,
                 head=2, min_group_planes=1, norm_layer_1d=nn.Identity,
                 **kwargs):
        super(RIM, self).__init__()

        self.transformer = Transformer(
            token_c, norm_layer_1d=norm_layer_1d, head=head,
            **kwargs)

        if input_dims == output_dims:
            self.feature_block = nn.Identity()  
        else:
            self.feature_block = nn.Sequential(
                    conv1x1_1d(input_dims, output_dims),
                    norm_layer_1d(output_dims)
                    )

        self.projectors = Projector(
                    token_c, output_dims, head=head,
                    min_group_planes=min_group_planes,
                    norm_layer_1d=norm_layer_1d)
        
        self.dynamic_f = nn.Sequential(
            conv1x1_1d(input_dims, token_c),
            norm_layer_1d(token_c),
            nn.ReLU(inplace=True),
            conv1x1_1d(token_c, token_c),
            norm_layer_1d(token_c)
            )

    def forward(self, in_feature, in_tokens, knn_idx):
        #in_feature: B, N, C
        #in_coords: B, N, 3
        B, L, K = knn_idx.shape
        B, C, N = in_feature.shape

        gather_fts = gather(
                in_feature, knn_idx.view(B, -1)
                ).view(B, -1, L, K)
       
        tokens = self.dynamic_f(gather_fts.max(dim=3)[0])
  
        t_c = tokens.shape[1]
        
        if in_tokens is not None:
            tokens += in_tokens
     
        tokens = self.transformer(tokens)

        out_feature = self.projectors(
                    self.feature_block(in_feature), tokens
                    ) 

        return out_feature, tokens

class RIM_ResidualBlock(nn.Module):
    def __init__(self, inc, outc, token_c, norm_layer_1d):
        super(RIM_ResidualBlock, self).__init__()
        if inc != outc:
            self.res_connect = nn.Sequential(
                    nn.Conv1d(inc, outc, 1),
                    norm_layer_1d(outc),
                    )
        else:
            self.res_connect = nn.Identity()
        self.vt1 = RIM(
               token_c, inc, inc, norm_layer_1d=norm_layer_1d)
        self.vt2 = RIM(
               token_c, inc, outc, norm_layer_1d=norm_layer_1d)

    def forward(self, inputs):
        in_feature, tokens, knn_idx = inputs    
        out, tokens = self.vt1(in_feature, tokens, knn_idx)
        out, tokens = self.vt2(out, tokens, knn_idx)

        return out, tokens

class YOGO(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        cr = kwargs['width_r']
        cs = [32, 64, 128, 256, 256]
        cs = [int(cr * x) for x in cs]

        self.token_l = kwargs['token_l']
        self.token_s = kwargs['token_s']
        self.token_c = kwargs['token_c']

        self.group_ = kwargs['group']
        self.ball_r = kwargs['ball_r']

        norm_layer = kwargs['norm']
        
        self.stem = nn.Sequential(
            conv1x1_1d(9, cs[0]),
            norm_layer(cs[0]),
            nn.ReLU(inplace=True),
            conv1x1_1d(cs[0], cs[0]),
            norm_layer(cs[0])
        )

        self.stage1 = nn.Sequential(
            RIM_ResidualBlock(cs[0], cs[1], token_c=self.token_c, norm_layer_1d=norm_layer),
        )

        self.stage2 = nn.Sequential(
            RIM_ResidualBlock(cs[1], cs[2], token_c=self.token_c, norm_layer_1d=norm_layer),
        )

        self.stage3 = nn.Sequential(
            RIM_ResidualBlock(cs[2], cs[3], token_c=self.token_c, norm_layer_1d=norm_layer),
        )

        self.stage4 = nn.Sequential(
            RIM_ResidualBlock(cs[3], cs[4], token_c=self.token_c, norm_layer_1d=norm_layer),
        ) 

        self.classifier = nn.Sequential(
            conv1x1_1d(cs[4], cs[4]),
            norm_layer(cs[4]),
            nn.ReLU(inplace=True),
            conv1x1_1d(cs[4], kwargs['num_classes']),
            )

    def forward(self, x):

        coords = x[:, :3, :]

        B, _, N = x.shape

        feature_stem = self.stem(x)

        center_pts = furthest_point_sample(
                coords, self.token_l)

        if self.group_ == 'ball_query':
            knn_idx = mf.ball_query(
                center_pts, coords, self.ball_r, self.token_s
                )
        else:
            knn_idx = knn_search(coords, center_pts, self.token_s)       
            knn_idx = torch.from_numpy(knn_idx).cuda()
        
        feature1, tokens = self.stage1((feature_stem, None, knn_idx))
              
        feature2, tokens = self.stage2((feature1, tokens, knn_idx))

        feature3, tokens = self.stage3((feature2, tokens, knn_idx))

        feature4, tokens = self.stage4((feature3, tokens, knn_idx))

        out = self.classifier(feature4)
        return out


