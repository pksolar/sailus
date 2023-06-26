import os, glob
import random

import torch, sys
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch.utils.data as data

def imgnorm(img):
    """
    采用什么归一化方式：
    1、除255
    2、0-255
    :param image:
    :return:
    normalize the input images
    """
    i_max = np.max(img)
    i_min = np.min(img)
    norm = (img-i_min)/(i_max-i_min)
    return norm
def cropimage(img_lumin,img_label,mask,l=256):
    """
    随机从图片中取 变成为l的正方形
    :param img: 亮度矩阵
    :return:
    """
    #随机取一个点：
    start_y = random.randint(0,2160-l)
    start_x = random.randint(0,4096-l)
    img_lumin = img_lumin[start_y:start_y+l,start_x:start_x+l]
    img_label = img_label[start_y:start_y+l,start_x:start_x+l]
    mask = mask[start_y:start_y+l,start_x:start_x+l]

    return img_lumin,img_label,mask



class Dataset_epoch(data.Dataset):
    def __init__(self,names,norm=False):
        #输入只读带有A的
        super(Dataset_epoch,self).__init__()
        self.names = names
        self.norm = norm
        #所有的数据都读到了index_pair里
        self.index_pair = names
    def __len__(self):
        return len(self.index_pair)

    def __getitem__(self, step):

        #把4个通道叠一起：
        img_pathA = self.index_pair[step]
        img_pathC = img_pathA.replace("A","C")
        img_pathG = img_pathA.replace("A","G")
        img_pathT = img_pathA.replace("A","T")

        label1_path = self.index_pair[step].repace("intensity","label").replace("A","label")
        #img1 = cv2.imread(img1_path,0)
        #label1 = cv2.imread(label1_path,0).astype('int64') #one hot 只能用LongTensor
        if self.norm == True:
            img1 = imgnorm(img1)
        img1_tensor = torch.from_numpy(img1).float()
        label1_tensor = torch.from_numpy(label1)
        img1_tensor = img1_tensor.unsqueeze(0)
        label1_tensor = torch.nn.functional.one_hot(label1_tensor)
        return img1_tensor,label1_tensor