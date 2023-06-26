import os, glob
import torch, sys
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch.utils.data as data

def imgnorm(img):
    """
    :param image:
    :return:
    normalize the input images
    """
    i_max = np.max(img)
    i_min = np.min(img)
    norm = (img-i_min)/(i_max-i_min)
    return norm

class Dataset_epoch(data.Dataset):
    def __init__(self,names,norm=False):
        super(Dataset_epoch,self).__init__()
        self.names = names
        self.norm = norm
        #所有的数据都读到了index_pair里
        self.index_pair = names
    def __len__(self):
        return len(self.index_pair)

    def __getitem__(self, step):
        img1_path = self.index_pair[step]
        label1_path = self.index_pair[step].replace('image','label')
        img1 = cv2.imread(img1_path,0)
        label1 = cv2.imread(label1_path,0).astype('int64') #one hot 只能用LongTensor


        if self.norm == True:
            img1 = imgnorm(img1)
        img1_tensor = torch.from_numpy(img1).float()
        label1_tensor = torch.from_numpy(label1)
        img1_tensor = img1_tensor.unsqueeze(0)
        label1_tensor = torch.nn.functional.one_hot(label1_tensor)
        return img1_tensor,label1_tensor