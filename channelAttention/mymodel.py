import torch
import torch.nn as nn

class DNA_Sequencer(nn.Module):
    def __init__(self):
        super(DNA_Sequencer, self).__init__()
        coresize = 7
        paddingsize = 3
        self.conv1 = nn.Conv2d(4, 32, (coresize, coresize), padding=paddingsize) #3个cycle,每个cycle 4张图
        self.conv2 = nn.Conv2d(32, 32, (coresize, coresize), padding=paddingsize)
        self.conv3 = nn.Conv2d(32, 32, (coresize, coresize), padding=paddingsize)
        self.conv4 = nn.Conv2d(32, 32, (coresize, coresize), padding=paddingsize)
        self.conv5 = nn.Conv2d(32, 4, (coresize, coresize), padding=paddingsize) #一次性预测3个label
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

class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # 压缩空间
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )


    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
class DNA_Sequencer_Atten(nn.Module):
    def __init__(self):
        super(DNA_Sequencer_Atten, self).__init__()
        coresize = 3
        paddingsize = 1
        self.conv1 = nn.Conv2d(4, 32, (coresize, coresize), padding=paddingsize) #3个cycle,每个cycle 4张图
        self.conv2 = nn.Conv2d(32, 32, (coresize, coresize), padding=paddingsize)
        self.conv3 = nn.Conv2d(32, 32, (coresize, coresize), padding=paddingsize)
        self.conv4 = nn.Conv2d(32, 32, (coresize, coresize), padding=paddingsize)
        self.conv5 = nn.Conv2d(32, 4, (coresize, coresize), padding=paddingsize) #一次性预测3个label
        self.relu = nn.ReLU()
        self.relu = SELayer(32)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        # x = self.se(x)
        x = self.relu(x)
        x = self.conv5(x)

        return x
