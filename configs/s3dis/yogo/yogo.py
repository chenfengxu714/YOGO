import torch.optim as optim

from models.s3dis import YOGO
from utils.config import Config, configs

# model
configs.model = Config(YOGO)
configs.model.num_classes = configs.data.num_classes
configs.model.token_l = 32
configs.model.token_s = 128
configs.model.token_c = 256
configs.model.group_ = 'ball_query'
configs.model.ball_r = 0.2

configs.model.extra_feature_channels = 6
configs.dataset.num_points = 4096

configs.train.optimizer.weight_decay = 0
# train: scheduler
configs.train.scheduler = Config(optim.lr_scheduler.CosineAnnealingLR)
configs.train.scheduler.T_max = configs.train.num_epochs
