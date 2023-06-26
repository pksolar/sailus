import glob
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.utils.data as data
import cv2
import os
import numpy as np
from natsort import natsorted

class Max(nn.Module):
    def __init__(self):
        super(Max, self).__init__()
    def forward(self,x): #输入是 b c h w

        return x

class DNA_Sequencer(nn.Module):
    """
    bn 有必要加，resnet有必要，cbam有必要，通道注意力有必要

    """
    def __init__(self):
        super(DNA_Sequencer, self).__init__()
        size = 7
        padding_size = int((size-1)/2)
        self.conv1 = nn.Conv2d(12, 32, 7, padding=3) #3个cycle,每个cycle 4张图
        self.conv2 = nn.Conv2d(32, 32, 7, padding=3)
        self.conv3 = nn.Conv2d(32, 32, size, padding=padding_size)
        self.conv4 = nn.Conv2d(32, 32, size, padding=padding_size)
        self.conv5 = nn.Conv2d(32, 4,size, padding=padding_size)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        return x

