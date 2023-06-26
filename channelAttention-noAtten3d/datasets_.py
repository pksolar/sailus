import os, glob
import random

import torch, sys
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch.utils.data as data


class Dataset_npy(data.Dataset):
    def __init__(self,data_array,label_array): # daa_array: pic_num,height,cycle,channel,  label_array: pic_num,height,cycle
        super(Dataset_npy,self).__init__()
        self.data_array = data_array
        self.lable_array = label_array
    def __len__(self):
        return len(self.data_array)

    def __getitem__(self, idx):#
        input_array = self.data_array[idx] # height,cyc,cha
        input_label = self.lable_array[idx]# height,cyc
        input_array = np.transpose(input_array,[2,0,1])[np.newaxis,:]
        #make array to tensor
        tensor_input = torch.from_numpy(input_array).float()
        label_tensor = torch.LongTensor(input_label)#
        label_tensor = torch.nn.functional.one_hot(label_tensor,6)[:,:,1:5].float() #
        label_tensor = label_tensor.permute(2,0,1).contiguous().unsqueeze(0)

        return tensor_input,label_tensor


class Dataset_npy_val(data.Dataset):
    def __init__(self,data_array,label_array): # daa_array: pic_num,height,cycle,channel,  label_array: pic_num,height,cycle
        super(Dataset_npy_val,self).__init__()
        self.data_array = data_array
        self.lable_array = label_array
    def __len__(self):
        return len(self.data_array)

    def __getitem__(self, idx):#
        input_array = self.data_array[idx]  # height,cyc,cha
        input_label = self.lable_array[idx]  # height,cyc
        input_array = np.transpose(input_array, [2, 0, 1])[np.newaxis, :]
        # make array to tensor
        tensor_input = torch.from_numpy(input_array).float()
        label_tensor = torch.LongTensor(input_label)  #
        label_tensor = torch.nn.functional.one_hot(label_tensor, 6)[:, :, 1:5].float()  #
        label_tensor = label_tensor.permute(2, 0, 1).contiguous().unsqueeze(0)

        return tensor_input,label_tensor