import glob
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.utils.data as data
import cv2
import os
import numpy as np
from natsort import natsorted


class DNA_Sequencer(nn.Module):
    def __init__(self):
        super(DNA_Sequencer, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, (3, 3), padding=1) #3个cycle,每个cycle 4张图
        self.conv2 = nn.Conv2d(32, 32, (3, 3), padding=1)
        self.conv3 = nn.Conv2d(32, 32, (3, 3), padding=1)
        self.conv4 = nn.Conv2d(32, 32, (3, 3), padding=1)
        self.conv5 = nn.Conv2d(32, 4, (3, 3), padding=1)
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
        return x

