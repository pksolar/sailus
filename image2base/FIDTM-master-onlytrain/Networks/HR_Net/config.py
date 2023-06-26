import os
from easydict import EasyDict as edict
import time
import torch

# init
__C = edict()
cfg = __C

#------------------------------TRAIN------------------------
__C.SEED = 1  # random seed,  for reproduction


__C.NET = 'HR_Net'
__C.PRE_HR_WEIGHTS = 'E:\code\python_PK\image2base\FIDTM-master\weight/hrnetv2_w48_imagenet_pretrained.pth'
