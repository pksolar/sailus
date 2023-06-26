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
class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=8, spatial_kernel=7):
        super(CBAMLayer, self).__init__()

        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return  x
class DNA_Sequencer(nn.Module):
    def __init__(self):
        super(DNA_Sequencer, self).__init__()
        self.conv1 = nn.Conv2d(12, 32, (7, 7), padding=3) #3个cycle,每个cycle 4张图
        self.conv2 = nn.Conv2d(32, 32, (7, 7), padding=3)
        self.conv3 = nn.Conv2d(32, 32, (7, 7), padding=3)
        self.conv4 = nn.Conv2d(32, 32, (7, 7), padding=3)
        self.conv5 = nn.Conv2d(32, 32, (7, 7), padding=3) #一次性预测3个label
        self.conv6_1 = nn.Conv2d(32, 4, (7, 7), padding=3) #一次性预测3个label
        self.conv6_2 = nn.Conv2d(32, 4, (7, 7), padding=3)  # 一次性预测3个label
        self.conv6_3 = nn.Conv2d(32, 4, (7, 7), padding=3)  # 一次性预测3个label
        self.relu = nn.ReLU()
        self.cbam = CBAMLayer(32)


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
        x = self.cbam(x)
        x = self.relu(x)
        x1 = self.conv6_1(x)
        x2 = self.conv6_2(x)
        x3 = self.conv6_3(x)
        x = torch.cat([x1,x2,x3],dim = 1)

        return x

