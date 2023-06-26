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
    img = np.concatenate([imgA,imgC,imgG,imgT]).astype(np.float32) #整数chu'fa
    img = norm_max_min(img)
    return img

# def norm(arr):# 对每个通道各自做归一化。
#     for i in range(arr.shape[0]):
#         channel = arr[i]
#         # print("norm again")
#         min_val = np.percentile(channel,1)
#         max_val = np.percentile(channel,99)
#         arr[i] = (channel - min_val) / (max_val - min_val)
#     return arr

def norm(arr):# 对每个通道各自做归一化。
    for i in range(arr.shape[0]):
        channel = arr[i]
        # print("norm again")
        min_val = np.min(channel)
        max_val = np.max(channel)
        arr[i] = (channel - min_val) / (max_val - min_val)
    return arr

def norm_max_min(arr):# 对每个通道各自做归一化。
    for i in range(arr.shape[0]):
        channel = arr[i]
        # print("norm again")
        min_val = np.min(channel)
        max_val = np.max(channel)
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


        #middle的时候：
        img_middle = read4Img(path)
        # np.save("debug/"+name_file,img_middle)
        c,w,d = img_middle.shape

        label_middle_path = path.replace("img", "label").replace("_" + basetype, ".npy").replace("08h","08husen")  # basetype: A.tif
        #print("label middle: ", label_middle_path)
        # name_file = name_file + " label:" + os.path.basename(label_path)
        label_middle = np.load(label_middle_path)

        # front的时候：
        cyc_f_number = cyc_m_number - 1
        img_front_path = path.replace(cycname, 'Cyc' + str(cyc_f_number).zfill(3))
        label_front_path = img_front_path.replace("img", "label").replace("_" + basetype, ".npy").replace("08h","08husen")
        #print("label front: ", label_front_path)

        try:
            img_front = read4Img(img_front_path)
            label_front = np.load(label_front_path)
        except:
            img_front = np.zeros((4, w, d)).astype(np.float32)
            label_front = np.zeros((w, d)).astype(int)



        # behind的时候：
        cyc_h_number = cyc_m_number + 1
        img_behind_path = path.replace(cycname, 'Cyc' + str(cyc_h_number).zfill(3))
        label_behind_path = img_behind_path.replace("img", "label").replace("_" + basetype, ".npy").replace("08h","08husen")
        #print("label behind: ",label_behind_path)

        try:
            img_behind = read4Img(img_behind_path)
            label_behind = np.load(label_behind_path)
        except:
            img_behind = np.zeros((4, w, d)).astype(np.float32)
            label_behind = np.zeros((w, d)).astype(int)


        #concate the three cycle img
        img = np.concatenate([img_front,img_middle,img_behind])

        label = np.concatenate([label_front[np.newaxis, :], label_middle[np.newaxis, :], label_behind[np.newaxis, :]])


        # np.save("debug/label_" + name_file, label)
        # 这个msk必须时已经经过膨胀处理的mask。
        msk_path = path.replace(os.path.basename(path), f"{machine}_{fov}_msk.npy").replace("img", "msk").replace("08h","08husen")
        #print("msk_path: ",msk_path)
        msk = np.load(msk_path)
        #name_file = name_file + " msk:"+os.path.basename(msk_path)
        msk[msk<0] = 0
        # np.save("debug/msk_" + name_file , msk)


        width = 256
        height = 256
        crop_size_x = random.randint(0, img.shape[1] - width)
        crop_size_y = random.randint(0, img.shape[2] - height)
        img = img[:, crop_size_x: crop_size_x + width, crop_size_y:crop_size_y + height].copy()
        label = label[:,crop_size_x: crop_size_x + width, crop_size_y:crop_size_y + height].copy()
        msk = msk[crop_size_x: crop_size_x + width, crop_size_y:crop_size_y + height].copy()


        img_tensor = torch.from_numpy(img).float()
        label_tensor = torch.LongTensor(label)  # 已经自己将其做成了one hot形式
        label_front_onehot = torch.nn.functional.one_hot(label_tensor[0], 6)[:, :, 1:5].float()  #
        label_middle_onehot = torch.nn.functional.one_hot(label_tensor[1], 6)[:, :, 1:5].float()  #
        label_behind_onehot = torch.nn.functional.one_hot(label_tensor[2], 6)[:, :, 1:5].float()  #
        label_tensor_onehot = torch.cat([label_front_onehot, label_middle_onehot, label_behind_onehot], dim=-1)
        # label_tensor = torch.nn.functional.one_hot(label_tensor,6)[:,:,1:5].float()#
        label_tensor_onehot = label_tensor_onehot.permute(2, 0, 1).contiguous()

        msk_tensor = torch.from_numpy(msk).unsqueeze(0).float()
        # numpy_label = label_tensor.numpy()
        return img_tensor,label_tensor_onehot,msk_tensor,name_file


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

        name_file = os.path.basename(path)
        name_list = os.path.basename(path).split("_")

        machine = name_list[0]
        fov = name_list[1]
        cycname = name_list[2]
        basetype = name_list[3]

        cyc_m_number = int(cycname[3:])


        #middle的时候：
        img_middle = read4Img(path)
        # np.save("debug/"+name_file,img_middle)
        c,w,d = img_middle.shape

        label_middle_path = path.replace("img", "label").replace("_" + basetype, ".npy")  # basetype: A.tif
        # name_file = name_file + " label:" + os.path.basename(label_path)
        label_middle = np.load(label_middle_path)

        # front的时候：
        cyc_f_number = cyc_m_number - 1
        img_front_path = path.replace(cycname, 'Cyc' + str(cyc_f_number).zfill(3))
        label_front_path = img_front_path.replace("img", "label").replace("_" + basetype, ".npy")

        try:
            img_front = read4Img(img_front_path)
            label_front = np.load(label_front_path)
        except:
            img_front = np.zeros((4, w, d))
            label_front = np.zeros((w, d))



        # behind的时候：
        cyc_h_number = cyc_m_number + 1
        img_behind_path = path.replace(cycname, 'Cyc' + str(cyc_h_number).zfill(3))
        label_behind_path = img_behind_path.replace("img", "label").replace("_" + basetype, ".npy")

        try:
            img_behind = read4Img(img_behind_path)
            label_behind = np.load(label_behind_path)
        except:
            img_behind = np.zeros((4, w, d))
            label_behind = np.zeros((w, d))


        #concate the three cycle img
        img = np.concatenate([img_front,img_middle,img_behind])

        label = np.concatenate([label_front[np.newaxis, :], label_middle[np.newaxis, :], label_behind[np.newaxis, :]])


        # np.save("debug/label_" + name_file, label)
        # 这个msk必须时已经经过膨胀处理的mask。
        msk_path = path.replace(os.path.basename(path), f"{machine}_{fov}_msk.npy").replace("img", "msk")
        msk = np.load(msk_path)
        #name_file = name_file + " msk:"+os.path.basename(msk_path)
        msk[msk<0] = 0
        # np.save("debug/msk_" + name_file , msk)



        width = 512
        height = 512
        crop_size_x = 1000
        crop_size_y = 1000
        img = img[:, crop_size_x: crop_size_x + width, crop_size_y:crop_size_y + height].copy()
        label = label[:,crop_size_x: crop_size_x + width, crop_size_y:crop_size_y + height].copy()
        msk = msk[crop_size_x: crop_size_x + width, crop_size_y:crop_size_y + height].copy()



        img_tensor = torch.from_numpy(img).float()
        label_tensor = torch.LongTensor(label)  # 已经自己将其做成了one hot形式
        label_front_onehot = torch.nn.functional.one_hot(label_tensor[0], 6)[:, :, 1:5].float()  #
        label_middle_onehot = torch.nn.functional.one_hot(label_tensor[1], 6)[:, :, 1:5].float()  #
        label_behind_onehot = torch.nn.functional.one_hot(label_tensor[2], 6)[:, :, 1:5].float()  #
        label_tensor_onehot = torch.cat([label_front_onehot, label_middle_onehot, label_behind_onehot], dim=-1)
        # label_tensor = torch.nn.functional.one_hot(label_tensor,6)[:,:,1:5].float()#
        label_tensor_onehot = label_tensor_onehot.permute(2, 0, 1).contiguous()

        msk_tensor = torch.from_numpy(msk).unsqueeze(0).float()
        # numpy_label = label_tensor.numpy()
        return img_tensor,label_tensor_onehot,msk_tensor



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

        path = self.index_pair[step]

        name_file = os.path.basename(path)
        name_list = os.path.basename(path).split("_")

        machine = name_list[0]
        fov = name_list[1]
        cycname = name_list[2]
        basetype = name_list[3]

        cyc_m_number = int(cycname[3:])

        # middle的时候：
        img_middle = read4Img(path)
        # np.save("debug/"+name_file,img_middle)
        c, w, d = img_middle.shape

        label_middle_path = path.replace("img", "label").replace("_" + basetype, ".npy")  # basetype: A.tif
        # name_file = name_file + " label:" + os.path.basename(label_path)
        label_middle = np.load(label_middle_path)

        # front的时候：
        cyc_f_number = cyc_m_number - 1
        img_front_path = path.replace(cycname, 'Cyc' + str(cyc_f_number).zfill(3))
        label_front_path = img_front_path.replace("img", "label").replace("_" + basetype, ".npy")

        try:
            img_front = read4Img(img_front_path)
            label_front = np.load(label_front_path)
        except:
            img_front = np.zeros((4, w, d))
            label_front = np.zeros((w, d))

        # behind的时候：
        cyc_h_number = cyc_m_number + 1
        img_behind_path = path.replace(cycname, 'Cyc' + str(cyc_h_number).zfill(3))
        label_behind_path = img_behind_path.replace("img", "label").replace("_" + basetype, ".npy")

        try:
            img_behind = read4Img(img_behind_path)
            label_behind = np.load(label_behind_path)
        except:
            img_behind = np.zeros((4, w, d))
            label_behind = np.zeros((w, d))

        # concate the three cycle img
        img = np.concatenate([img_front, img_middle, img_behind])

        label = np.concatenate([label_front[np.newaxis, :], label_middle[np.newaxis, :], label_behind[np.newaxis, :]])

        # np.save("debug/label_" + name_file, label)
        # 这个msk必须时已经经过膨胀处理的mask。
        msk_path = path.replace(os.path.basename(path), f"{machine}_{fov}_msk.npy").replace("img", "msk")
        msk = np.load(msk_path)
        # name_file = name_file + " msk:"+os.path.basename(msk_path)
        # np.save("debug/msk_" + name_file , msk)
        img = img.copy()
        msk = abs(msk)

        pad_width_img = ((0, 0), (2, 2), (0, 0))
        pad_width = ((2, 2), (0, 0))
        img = np.pad(img, pad_width_img, mode='constant')
        label = np.pad(label,pad_width_img,mode='constant')
        msk = np.pad(msk,pad_width,mode='constant')
        # msk_sum = sum(sum(msk))
        # print(msk_sum)

        img_tensor = torch.from_numpy(img).float()
        label_tensor = torch.LongTensor(label)  # 已经自己将其做成了one hot形式
        label_front_onehot = torch.nn.functional.one_hot(label_tensor[0], 6)[:, :, 1:5].float()  #
        label_middle_onehot = torch.nn.functional.one_hot(label_tensor[1], 6)[:, :, 1:5].float()  #
        label_behind_onehot = torch.nn.functional.one_hot(label_tensor[2], 6)[:, :, 1:5].float()  #
        label_tensor_onehot = torch.cat([label_front_onehot, label_middle_onehot, label_behind_onehot], dim=-1)
        # label_tensor = torch.nn.functional.one_hot(label_tensor,6)[:,:,1:5].float()#
        label_tensor_onehot = label_tensor_onehot.permute(2, 0, 1).contiguous()

        msk_tensor = torch.from_numpy(msk).unsqueeze(0).float()
        # numpy_label = label_tensor.numpy()
        return img_tensor, label_tensor_onehot, msk_tensor
