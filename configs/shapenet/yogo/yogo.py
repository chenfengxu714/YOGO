import torch.optim as optim
import torch.nn as nn
from models.shapenet import YOGO
from utils.config import Config, configs

# model
configs.model = Config(YOGO)
configs.model.num_classes = configs.data.num_classes

configs.model.num_shapes = configs.data.num_shapes
configs.model.extra_feature_channels = 3
configs.model.token_l = 32
configs.model.token_s = 96
configs.model.token_c = 256
configs.model.group = 'ball_query'
configs.model.ball_r = 0.2
configs.model.width_r = 1
configs.model.norm = nn.InstanceNorm1d

configs.train.num_epochs = 250
configs.train.scheduler = Config(optim.lr_scheduler.CosineAnnealingLR)
configs.train.scheduler.T_max = configs.train.num_epochs
