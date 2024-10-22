import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
#import torchpack.distributed as dist

__all__ = ['cosine_schedule_with_warmup']
dist = 1
def cosine_schedule_with_warmup(k, num_epochs, batch_size, dataset_size=57216):
    batch_size *= dist.size()
    if dist.size() == 1:
        warmup_iters = 0
    else:
        warmup_iters = 1000 // dist.size()


    if k < warmup_iters:
        return (k + 1) / warmup_iters
    else:
        iter_per_epoch = (dataset_size + batch_size - 1) // batch_size
        return 0.5 * (1 + np.cos(np.pi * (k - warmup_iters) /
                                (num_epochs * iter_per_epoch)))
