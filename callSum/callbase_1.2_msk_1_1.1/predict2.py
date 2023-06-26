import glob
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.utils.data as data
import cv2
import os
import numpy as np
from natsort import natsorted
from data.datasets import Dataset_epoch_test
from callNet.model1 import  DNA_Sequencer
from utils.utils import AverageMeter

dataset_name = "30"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCHSIZE = 1
N_CLASS = 4
classes = ['N', 'A', 'C', 'G', 'T']  # n0,a1,c2,g3,t4
test_dir_total = glob.glob("E:\code\python_PK\callbase\datasets\{}\Res\Lane01\deepLearnData\*\intensity_norm\*_A.npy".format(dataset_name)) #total_dir = glob.glob(r"E:\code\python_PK\callbase\datasets\{}\Res\Lane01\deepLearnData\*\intensity_norm\*_A.npy".format(dataset_name))

test_dir = test_dir_total[int(0.4*(len(test_dir_total))):]
test_dataset = Dataset_epoch_test(test_dir)
test_loader = data.DataLoader(test_dataset, batch_size=BATCHSIZE, shuffle=False)




model = DNA_Sequencer().to(device)

best_model = torch.load("savedir_pth/accbest.pth.tar")['state_dict']
model.load_state_dict(best_model)




idx = 0
with torch.no_grad():
    for inputs, labels, msk in test_loader:  #
        print("path: ",test_dir[idx])
        idx  = idx +1
        model.eval()
        inputs = inputs.to(device)
        labels = labels.to(device)  # labels size;b,c,h,w
        msk = msk.to(device)
        outputs = model(inputs)
        #loss_val = criterion(outputs * msk, labels * msk)
        # print("loss_val:", loss_val.item())
        # loss_val_show.update(loss_val.item())
        # 只统计msk的位置。

        # 只比msk的位置。
        _, preds = torch.max(outputs,
                             1)  # preds size [b,h,w],值是0，1,2,3，label是0,1,,,哪个通道的概率值最大，输出是,第几个通道的值最大。，b is batchsize,h,w is the predicted result.save it.
        _, label_ = torch.max(labels, 1)  # label的值也是0,1,2,3，
        # 用mask把不必要的不去比较。mask是，0,1 变成，（-10,0）,然后和label以及preds相加。得到preds和label是负值的地方就是背景，拍平成列表，再删除也元素，得到很短的列表，再去比较。速度会大大加快。

        c = (preds == label_)  # 里面有多少true和false，

        msk = torch.squeeze(abs(msk)) #msk b,1,b,w
        total = torch.sum(msk).item()
        right = torch.sum(msk* c).item()
        accurate = 100 * right/total
        # # 这个统计好好写。
        # # 展平c和label
        # c = c.flatten(0)  # 是不是不必拍平。
        # label_ = label_.flatten(0)
        # msk = msk.flatten(0)
        # 只比有mask的位置，也只统计这里的位置。
        # accurate = 100 * torch.sum(class_correct) / torch.sum(class_total)
        # accurate_show.update(accurate.item())
        print("total accuray: %.2f %%" % (accurate))

