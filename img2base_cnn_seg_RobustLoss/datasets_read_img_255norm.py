import os, glob
import random

import torch, sys
from torch.utils.data import Dataset
# import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch.utils.data as data

def read4Img(path):
    name_path_C = path.replace("_A", '_C')
    name_path_G = path.replace("_A", '_G')
    name_path_T = path.replace("_A", '_T')

    imgA = cv2.imread(path, 0)[np.newaxis,:]
    imgC = cv2.imread(name_path_C, 0)[np.newaxis,:]
    imgG = cv2.imread(name_path_G, 0)[np.newaxis,:]
    imgT = cv2.imread(name_path_T, 0)[np.newaxis,:]
    img = np.concatenate([imgA,imgC,imgG,imgT]).astype(np.float64) #整数chu'fa
    img = img /255.0
    return img




def norm(arr):# 对每个通道各自做归一化。
    for i in range(arr.shape[0]):
        channel = arr[i]
        # print("norm again")
        min_val = np.percentile(channel,1)
        max_val = np.percentile(channel,99)
        arr[i] = (channel - min_val) / (max_val - min_val)
    return arr

class Dataset_3cycle(data.Dataset):
    """
     r"E:\data\deep\image2base\image\image\*.tif")z
    """
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

        name_file = os.path.basename(path)
        name_list = os.path.basename(path).split("_")

        machine = name_list[0]
        fov = name_list[1]
        cycname = name_list[2]
        basetype = name_list[3]

        cyc_m_number = int(cycname[3:])



        img_middle = read4Img(path)
        # np.save("debug/"+name_file,img_middle)
        c,w,d = img_middle.shape

        if cycname == 'Cyc001': #前一个为全0矩阵
            img_front = np.zeros((4,w,d))
        else:
            cyc_f_number = cyc_m_number - 1
            img_front_path = path.replace(cycname,'Cyc'+str(cyc_f_number).zfill(3))
            img_front = read4Img(img_front_path)

        if cycname == 'Cyc100':
            img_behind = np.zeros((4,w,d))
        else:
            cyc_h_number = cyc_m_number + 1
            img_behind_path = path.replace(cycname, 'Cyc' + str(cyc_h_number).zfill(3))
            img_behind = read4Img(img_behind_path)

        #concate the three cycle img
        img = np.concatenate([img_front,img_middle,img_behind])

        label_path = path.replace("img", "label").replace(basetype+"_","").replace("tif","npy")
        #name_file = name_file + " label:" + os.path.basename(label_path)
        label = np.load(label_path)
        # np.save("debug/label_" + name_file, label)
        # 这个msk必须时已经经过膨胀处理的mask。
        msk_path = path.replace(os.path.basename(path), f"{machine}_{fov}_msk.npy").replace("img", "msk")
        msk = np.load(msk_path)
        #name_file = name_file + " msk:"+os.path.basename(msk_path)
        msk[msk<0] = 0
        # np.save("debug/msk_" + name_file , msk)

        "data augmention"
        # if random.random() > 0.5:
        #     label = np.fliplr(label)
        #     msk = np.fliplr(msk)
        #     img = np.fliplr(img)
        #
        #     np.save("debug/label_" + name_file, label)
        #     np.save("debug/msk_" + name_file, msk)
        #     np.save("debug/img_" + name_file, img)


        width = 256
        height = 256
        crop_size_x = random.randint(0, img.shape[1] - width)
        crop_size_y = random.randint(0, img.shape[2] - height)
        img = img[:, crop_size_x: crop_size_x + width, crop_size_y:crop_size_y + height].copy()
        label = label[crop_size_x: crop_size_x + width, crop_size_y:crop_size_y + height].copy()
        msk = msk[crop_size_x: crop_size_x + width, crop_size_y:crop_size_y + height].copy()


        img_tensor = torch.from_numpy(img).float()
        label_tensor = torch.LongTensor(label) #已经自己将其做成了one hot形式
        label_tensor = torch.nn.functional.one_hot(label_tensor,6)[:,:,1:5].float()#
        label_tensor = label_tensor.permute(2, 0, 1).contiguous()
        msk_tensor = torch.from_numpy(msk).unsqueeze(0).float()
        numpy_label = label_tensor.numpy()



        return img_tensor,label_tensor,msk_tensor,name_file


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
        name_list = os.path.basename(path).split("_")
        machine = name_list[0]
        fov = name_list[1]
        cycname = name_list[2]
        basetype = name_list[3]

        cyc_m_number = int(cycname[3:])

        img_middle = read4Img(path)
        c, w, d = img_middle.shape

        if cycname == 'Cyc001':  # 前一个为全0矩阵
            img_front = np.zeros((4, w, d))
        else:
            cyc_f_number = cyc_m_number - 1
            img_front_path = path.replace(cycname, 'Cyc' + str(cyc_f_number).zfill(3))
            img_front = read4Img(img_front_path)

        if cycname == 'Cyc100':
            img_behind = np.zeros((4, w, d))
        else:
            cyc_h_number = cyc_m_number + 1
            img_behind_path = path.replace(cycname, 'Cyc' + str(cyc_h_number).zfill(3))
            img_behind = read4Img(img_behind_path)

        # concate the three cycle img
        img = np.concatenate([img_front, img_middle, img_behind]).copy()

        label_path = path.replace("img", "label").replace(basetype + "_", "").replace("tif", "npy")
        label = np.load(label_path)

        # 这个msk必须时已经经过膨胀处理的mask。
        msk_path = path.replace(os.path.basename(path), f"{machine}_{fov}_msk.npy").replace("img", "msk")
        msk = np.load(msk_path)
        msk[msk<0] = 0

        width = 512
        height = 512
        crop_size_x = 1000
        crop_size_y = 1000
        img = img[:, crop_size_x: crop_size_x + width, crop_size_y:crop_size_y + height].copy()
        label = label[crop_size_x: crop_size_x + width, crop_size_y:crop_size_y + height].copy()
        msk = msk[crop_size_x: crop_size_x + width, crop_size_y:crop_size_y + height].copy()

        img_tensor = torch.from_numpy(img).float()
        label_tensor = torch.LongTensor(label)  # 已经自己将其做成了one hot形式
        label_tensor = torch.nn.functional.one_hot(label_tensor, 6)[:, :, 1:5].float()
        label_tensor = label_tensor.permute(2, 0, 1).contiguous()
        msk_tensor = torch.from_numpy(msk).unsqueeze(0).float()

        return img_tensor, label_tensor, msk_tensor



class Dataset_3cycle_test(data.Dataset):
    def __init__(self, names, norm=False):
        # 输入只读带有A的
        super(Dataset_3cycle_test, self).__init__()
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
        name_list = os.path.basename(path).split("_")
        machine = name_list[0]
        fov = name_list[1]
        cycname = name_list[2]
        basetype = name_list[3]


        cyc_m_number = int(cycname[3:])

        img_middle = read4Img(path)
        c, w, d = img_middle.shape

        if cycname == 'Cyc001':  # 前一个为全0矩阵
            img_front = np.zeros((4, w, d))
        else:
            cyc_f_number = cyc_m_number - 1
            img_front_path = path.replace(cycname, 'Cyc' + str(cyc_f_number).zfill(3))
            img_front = read4Img(img_front_path)

        if cycname == 'Cyc100':
            img_behind = np.zeros((4, w, d))
        else:
            cyc_h_number = cyc_m_number + 1
            img_behind_path = path.replace(cycname, 'Cyc' + str(cyc_h_number).zfill(3))
            img_behind = read4Img(img_behind_path)

        # concate the three cycle img
        img = np.concatenate([img_front, img_middle, img_behind])

        label_path = path.replace("img", "label").replace(basetype + "_", "").replace("tif", "npy")
        label = np.load(label_path)

        # 这个msk必须时已经经过膨胀处理的mask。
        msk_path = path.replace(os.path.basename(path), f"{machine}_{fov}_msk.npy").replace("img", "msk")
        msk = np.load(msk_path)
        # msk[msk<0] = 0
        img = img.copy()
        msk = abs(msk)


        img_tensor = torch.from_numpy(img).float()
        label_tensor = torch.LongTensor(label)  # 已经自己将其做成了one hot形式
        label_tensor = torch.nn.functional.one_hot(label_tensor, 6)[:, :, 1:5].float()
        label_tensor = label_tensor.permute(2, 0, 1).contiguous()
        msk_tensor = torch.from_numpy(msk).unsqueeze(0).float()

        return img_tensor, label_tensor, msk_tensor