import os, glob
import random

import torch, sys
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch.utils.data as data


def listToTensor(list):
    tensor = torch.empty(list.__len__(), list[0].__len__())
    for i in range(list.__len__()):
        tensor[i, :] = torch.FloatTensor(list[i])
    return tensor #二维tensor

class Dataset_npy(data.Dataset):
    def __init__(self,data_array,label_array,seq):
        super(Dataset_npy,self).__init__()
        # self.names = names
        # self.norm = norm
        #所有的数据都读到了index_pair里
        self.index_pair = data_array
        self.seq = seq
        self.lable_array = label_array
    def __len__(self):
        return len(self.index_pair)

    def __getitem__(self, idx):#
        #两个选择：
        #完全随机。
        cycle_idx = random.randint(0,100)
        # print("cycle_idx: ",cycle_idx)
        # print("idx:",idx)
        #data_1 = self.index_pair[idx][cycle_idx] # idx 在train数据里是随机取得第n个点的数据。
        if cycle_idx + self.seq < 100:
            data_input = self.index_pair[idx][cycle_idx:cycle_idx+self.seq]  # idx 在train数据里是随机取得第n个点的数据。 shape:  :3,4
            label = self.lable_array[idx][cycle_idx:cycle_idx+self.seq] # shape : 3,
        else:
            data_input = np.zeros([self.seq,4])
            label = np.zeros([self.seq])
            data_input[:100-cycle_idx] = self.index_pair[idx][cycle_idx:]
            label[:100-cycle_idx] = self.lable_array[idx][cycle_idx:]


        #make array to tensor
        tensor_input = torch.from_numpy(data_input).float()
        label_tensor = torch.LongTensor(label)
        label_tensor = torch.nn.functional.one_hot(label_tensor,6)[:,1:5].float()

        return tensor_input,label_tensor


class Dataset_npy2(data.Dataset):
    def __init__(self,data_array,label_array,seq):
        super(Dataset_npy2,self).__init__()
        # self.names = names
        # self.norm = norm
        #所有的数据都读到了index_pair里
        self.index_pair = data_array
        self.seq = seq
        self.lable_array = label_array
    def __len__(self):
        return len(self.index_pair)

    def __getitem__(self, idx):#
        #两个选择：
        #完全随机。
        cycle_idx = random.randint(0,100)
        # print("cycle_idx: ",cycle_idx)
        # print("idx:",idx)
        #data_1 = self.index_pair[idx][cycle_idx] # idx 在train数据里是随机取得第n个点的数据。
        if cycle_idx + self.seq < 100:
            data_input = self.index_pair[idx][cycle_idx:cycle_idx+self.seq]  # idx 在train数据里是随机取得第n个点的数据。 shape:  :3,4
            label = self.lable_array[idx][cycle_idx:cycle_idx+self.seq] # shape : 3,
        else:
            data_input = np.zeros([self.seq,4])
            label = np.zeros([self.seq])
            data_input[:100-cycle_idx] = self.index_pair[idx][cycle_idx:]
            label[:100-cycle_idx] = self.lable_array[idx][cycle_idx:]


        #make array to tensor
        tensor_input = torch.from_numpy(data_input).float()
        label_tensor = torch.LongTensor(label)
        label_tensor = torch.nn.functional.one_hot(label_tensor,6)[:,1:5].float()

        return tensor_input,label_tensor









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
        img_pathA = self.index_pair[step]
        img_pathC = img_pathA.replace("A", "C")
        img_pathG = img_pathA.replace("A", "G")
        img_pathT = img_pathA.replace("A", "T")

        dataset_name = img_pathA.split("\\")[5]

        label_path = img_pathA.replace("intensity_norm", "label").replace("A", "label")
        mask_path_dir = r"E:\code\python_PK\callbase\datasets\{}\Res\Lane01\deepLearnData\\".format(dataset_name)
        mask_path = mask_path_dir + img_pathA.split("\\")[-1].replace("A", "mask")

        mask = np.load(mask_path)
        imgA = np.load(img_pathA)
        imgC = np.load(img_pathC)
        imgG = np.load(img_pathG)
        imgT = np.load(img_pathT)
        label = np.load(label_path)

        img = np.concatenate((imgA[np.newaxis, :],imgC[np.newaxis, :],imgG[np.newaxis, :],imgT[np.newaxis, :]))

        label_ = label.copy()
        label_[label_!=5] = 1
        label_[label_==5] = 0

        mask[mask==-1] = 0 #大部分都是对的，因此，可以把这句话注释了。



        img = np.multiply(img,label_)
        img_tensor = torch.from_numpy(img).float()
        label_tensor = torch.LongTensor(label)
        label_tensor = torch.nn.functional.one_hot(label_tensor,6)[:,:,1:5].float().permute(2, 0, 1).contiguous()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)




        return img_tensor[:,2,70],label_tensor[:,2,70]





