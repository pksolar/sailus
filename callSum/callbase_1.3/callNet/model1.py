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
    def __init__(self):
        super(DNA_Sequencer, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, (11, 11), padding=5) #3个cycle,每个cycle 4张图
        self.conv2 = nn.Conv2d(32, 64, (7, 7), padding=3)
        self.conv3 = nn.Conv2d(64, 128, (7, 7), padding=3)
        self.conv4 = nn.Conv2d(128, 64, (3, 3), padding=1)
        self.conv5 = nn.Conv2d(64, 32, (3, 3), padding=1)
        self.conv6 = nn.Conv2d(32, 4, (3, 3), padding=1)
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
        x = self.relu(x)
        x = self.conv6(x)
        return x

