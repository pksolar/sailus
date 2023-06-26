import glob
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.utils.data as data
import cv2
import os
import numpy as np
from natsort import natsorted
import torchvision

class Max(nn.Module):
    def __init__(self):
        super(Max, self).__init__()
    def forward(self,x): #输入是 b c h w

        return x

#以下是采用了预训练权重的unet。

class DownBlock(nn.Module):
    def __init__(self, num_convs, inchannels, outchannels,ksize=3,psize=1, pool=True):
        super(DownBlock, self).__init__()
        blk = []
        if pool:
            blk.append(nn.MaxPool2d(kernel_size=2, stride=2))
        for i in range(num_convs):
            if i == 0:
                blk.append(nn.Conv2d(inchannels, outchannels, kernel_size=ksize, padding=psize))
            else:
                blk.append(nn.Conv2d(outchannels, outchannels, kernel_size=3, padding=1))
            blk.append(nn.BatchNorm2d(outchannels))
            blk.append(nn.ReLU(inplace=True))
        self.layer = nn.Sequential(*blk)

    def forward(self, x):
            return self.layer(x)



class UpBlock(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(UpBlock, self).__init__()
        self.convt = nn.ConvTranspose2d(inchannels, outchannels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(inchannels, outchannels, kernel_size=3, padding=1),
            nn.BatchNorm2d(outchannels),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannels, outchannels, kernel_size=3, padding=1),
            nn.BatchNorm2d(outchannels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.convt(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class UNet_full(nn.Module):
    def __init__(self, nchannels=12, nclasses=4):
        super(UNet_full, self).__init__()
        self.down1 = DownBlock(2, nchannels, 64,ksize=7,psize=3, pool=False)
        self.down2 = DownBlock(2, 64, 128)
        self.down3 = DownBlock(2, 128, 256)
        self.down4 = DownBlock(2, 256, 512 )
        self.down5 = DownBlock(2, 512, 1024)
        self.up1 = UpBlock(1024, 512)
        self.up2 = UpBlock(512, 256)
        self.up3 = UpBlock(256, 128)
        self.up4 = UpBlock(128, 64)
        self.out = nn.Sequential(
            nn.Conv2d(64, nclasses, kernel_size=1)
        )

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.out(x)
class UNet(nn.Module):
    def __init__(self, nchannels=12, nclasses=4):
        super(UNet, self).__init__()
        self.down1 = DownBlock(2, nchannels, 32,ksize=7,psize=3, pool=False)
        self.down2 = DownBlock(2, 32, 64)
        self.down3 = DownBlock(2, 64, 128)
        self.down4 = DownBlock(2, 128, 256)
        self.down5 = DownBlock(2, 256, 512)
        self.up1 = UpBlock(512, 256)
        self.up2 = UpBlock(256, 128)
        self.up3 = UpBlock(128, 64)
        self.up4 = UpBlock(64, 32)
        self.out = nn.Sequential(
            nn.Conv2d(32, nclasses, kernel_size=1)
        )

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.out(x)
class UNet(nn.Module):
    def __init__(self, nchannels=12, nclasses=4):
        super(UNet, self).__init__()
        self.down1 = DownBlock(2, nchannels, 32,ksize=7,psize=3, pool=False)
        self.down2 = DownBlock(2, 32, 64)
        self.down3 = DownBlock(2, 64, 128)
        self.down4 = DownBlock(2, 128, 256)
        self.down5 = DownBlock(2, 256, 512)
        self.up1 = UpBlock(512, 256)
        self.up2 = UpBlock(256, 128)
        self.up3 = UpBlock(128, 64)
        self.up4 = UpBlock(64, 32)
        self.out = nn.Sequential(
            nn.Conv2d(32, nclasses, kernel_size=1)
        )

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.out(x)
#以下是采用了预训练权重的unet。


class UNet_3(nn.Module):
    def __init__(self, nchannels=12, nclasses=4):
        super(UNet_3, self).__init__()
        self.down1 = DownBlock(2, nchannels, 32,ksize=7,psize=3, pool=False)
        self.down2 = DownBlock(2, 32, 64)
        self.down3 = DownBlock(2, 64, 128)
        self.down4 = DownBlock(2, 128, 256)
        # self.down5 = DownBlock(2, 256, 512)
        self.up1 = UpBlock(256, 128)
        self.up2 = UpBlock(128, 64)
        self.up3 = UpBlock(64, 32)
        # self.up4 = UpBlock(64, 32)
        self.out = nn.Sequential(
            nn.Conv2d(32, nclasses, kernel_size=1)
        )

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        # x5 = self.down5(x4) 4) 4)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        return self.out(x)

class Decoder(nn.Module):
  def __init__(self, in_channels, middle_channels, out_channels):
    super(Decoder, self).__init__()
    self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    self.conv_relu = nn.Sequential(
        nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
        nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
        )
  def forward(self, x1, x2):
    x1 = self.up(x1)
    x1 = torch.cat((x1, x2), dim=1)
    x1 = self.conv_relu(x1)
    return x1

class Unet2(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        self.base_model = torchvision.models.resnet18(True)
        self.base_layers = list(self.base_model.children())
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            self.base_layers[1],
            self.base_layers[2])
        self.layer2 = nn.Sequential(*self.base_layers[3:5])
        self.layer3 = self.base_layers[5]
        self.layer4 = self.base_layers[6]
        self.layer5 = self.base_layers[7]
        self.decode4 = Decoder(512, 25656, 256)
        self.decode3 = Decoder(256, 256+128, 256)
        self.decode2 = Decoder(256, 128+64, 128)
        self.decode1 = Decoder(128, 64+64, 64)
        self.decode0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
            )
        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, input):
        e1 = self.layer1(input) # 64,128,128
        e2 = self.layer2(e1) # 64,64,64
        e3 = self.layer3(e2) # 128,32,32
        e4 = self.layer4(e3) # 256,16,16
        f = self.layer5(e4) # 512,8,8
        d4 = self.decode4(f, e4) # 256,16,16
        d3 = self.decode3(d4, e3) # 256,32,32
        d2 = self.decode2(d3, e2) # 128,64,64
        d1 = self.decode1(d2, e1) # 64,128,128
        d0 = self.decode0(d1) # 64,256,256
        out = self.conv_last(d0) # 1,256,256
        return out
