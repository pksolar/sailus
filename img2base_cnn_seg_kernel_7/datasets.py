import os, glob
import random

import torch, sys
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch.utils.data as data


def norm(arr):# 对每个通道各自做归一化。
    for i in range(arr.shape[0]):
        channel = arr[i]
        print("norm again")
        min_val = np.amin(channel)
        max_val = np.amax(channel)
        arr[i] = (channel - min_val) /  (max_val - min_val)
    return arr

class Dataset_3cycle(data.Dataset):
    def __init__(self,names,norm=False):
        #输入只读带有A的
        super(Dataset_3cycle,self).__init__()
        self.names = names
        self.norm = norm
        #所有的数据都读到了index_pair里
        self.index_pair = names
    def __len__(self):
        return len(self.index_pair)

    def __getitem__(self, step):
        """
        直接读取npy数据，将前后cycle的拼接。如果时第一个cycle那就拼0
        """
        path = self.index_pair[step]

        cycname = os.path.basename(path)[:6] #
        cyc_m_number = int(cycname[3:])

        img_middle = np.load(path)
        c,w,d = img_middle.shape

        if cycname == 'Cyc001': #前一个为全0矩阵
            img_front = np.zeros((4,w,d))
        else:
            cyc_f_number = cyc_m_number - 1
            img_front_path = path.replace(cycname,'Cyc'+str(cyc_f_number).zfill(3))
            img_front = np.load(img_front_path)

        if cycname == 'Cyc100':
            img_behind = np.zeros((4,w,d))
        else:
            cyc_h_number = cyc_m_number + 1
            img_behind_path = path.replace(cycname, 'Cyc' + str(cyc_h_number).zfill(3))
            img_behind = np.load(img_behind_path)

        #concate the three cycle img
        img = np.concatenate([img_front,img_middle,img_behind])

        label_path = path.replace("imgdata","label2")
        label = np.load(label_path)

        #这个msk必须时已经经过膨胀处理的mask。
        msk = np.load(path.replace("imgdata","mask").replace(os.path.basename(path),'R001C001_val.npy'))
        msk[msk<0] = 0

        "data augmention"
        if random.random() > 0.5:
            label = np.fliplr(label)
            msk = np.fliplr(msk)
            img = np.fliplr(img)

        width = 256
        height = 256
        crop_size_x = random.randint(0, img.shape[1] - width)
        crop_size_y = random.randint(0, img.shape[2] - height)
        img = img[:, crop_size_x: crop_size_x + width, crop_size_y:crop_size_y + height].copy()
        label = label[:, crop_size_x: crop_size_x + width, crop_size_y:crop_size_y + height].copy()
        msk = msk[crop_size_x: crop_size_x + width, crop_size_y:crop_size_y + height].copy()

        img_tensor = torch.from_numpy(img).float()
        label_tensor = torch.from_numpy(label).float() #已经自己将其做成了one hot形式
        msk_tensor = torch.from_numpy(msk).unsqueeze(0).float()




        return img_tensor,label_tensor,msk_tensor


class Dataset_3cycle_val(data.Dataset):
    def __init__(self, names, norm=False):
        # 输入只读带有A的
        super(Dataset_3cycle_val, self).__init__()
        self.names = names
        self.norm = norm
        # 所有的数据都读到了index_pair里
        self.index_pair = names

    def __len__(self):
        return len(self.index_pair)

    def __getitem__(self, step):
        """
        直接读取npy数据，将前后cycle的拼接。如果时第一个cycle那就拼0
        """
        path = self.index_pair[step]

        cycname = os.path.basename(path)[:6]  #
        cyc_m_number = int(cycname[3:])

        img_middle = np.load(path)
        c, w, d = img_middle.shape

        if cycname == 'Cyc001':  # 前一个为全0矩阵
            img_front = np.zeros(4, w, d)
        else:
            cyc_f_number = cyc_m_number - 1
            img_front_path = path.replace(cycname, 'Cyc' + str(cyc_f_number).zfill(3))
            img_front = np.load(img_front_path)

        if cycname == 'Cyc100':
            img_behind = np.zeros(4, w, d)
        else:
            cyc_h_number = cyc_m_number + 1
            img_behind_path = path.replace(cycname, 'Cyc' + str(cyc_h_number).zfill(3))
            img_behind = np.load(img_behind_path)

        # concate the three cycle img
        img = np.concatenate([img_front, img_middle, img_behind]).copy()

        label_path = path.replace("imgdata", "label2")
        label = np.load(label_path)

        # 这个msk必须时已经经过膨胀处理的mask。
        msk = np.load(path.replace("imgdata", "mask").replace(os.path.basename(path), 'R001C001_val.npy'))
        msk[msk<0] = 0
        img_tensor = torch.from_numpy(img).float()
        label_tensor = torch.from_numpy(label).float()  # 已经自己将其做成了one hot形式
        msk_tensor = torch.from_numpy(msk).unsqueeze(0).float()

        return img_tensor, label_tensor, msk_tensor
