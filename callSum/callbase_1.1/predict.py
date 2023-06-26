import glob
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.utils.data as data
import cv2
import os
import numpy as np
from natsort import natsorted
from data.datasets import Dataset_epoch
from callNet.model1 import  DNA_Sequencer
from utils.utils import AverageMeter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCHSIZE = 8
N_CLASS = 5
classes = ['N', 'A', 'C', 'G', 'T']  # n0,a1,c2,g3,t4
test_dir = glob.glob("data/image/train/*.jpg")
test_dataset = Dataset_epoch(test_dir)
test_loader = data.DataLoader(test_dataset, batch_size=BATCHSIZE, shuffle=False)

model = DNA_Sequencer().to(device)
accurate_test = AverageMeter()
with torch.no_grad:
    for inputs, labels in test_loader:
        model.eval()

        inputs = inputs.to(device)
        labels = labels.to(device)  # labels size;b,c,h,w
        outputs = model(inputs)

        # 验证的精度:
        class_correct = torch.zeros(N_CLASS)
        class_total = torch.zeros(N_CLASS)
        _, preds = torch.max(outputs,1)  # preds size [b,h,w],哪个通道的概率值最大，输出是,第几个通道的值最大。，b is batchsize,h,w is the predicted result.save it.
        c = (preds == labels)  #
        # 展平c和label
        c = c.flatten(0)
        labels = labels.flatten(0)

        for i in range(c.shape[0]):  # labels:  b c h w
            label = labels[i]  # label can only be 0-4
            class_correct[label] += c[i].item()
            class_total[label] += 1

        for i in range(N_CLASS):
            accurate = 100 * class_correct[i] / class_total[i]
            accurate_test.update(accurate)
            print('Accuracy of %3s:%.2f %%' % (classes[i], accurate))
        print("this batch accuray is : %.2f %%" % (100 * torch.sum(class_correct) / torch.sum(class_total)))
    print("total acc:")

