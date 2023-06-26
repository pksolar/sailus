import os, glob
import random

import torch, sys
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch.utils.data as data

def imgnorm(img,p1,p99):
    """
    采用什么归一化方式：
    1、除255
    2、0-255
    :param image:
    :return:
    normalize the input images
    """
    norm = (img-p1)/(p99-p1)
    return norm
def cropimage(img_lumin_list,img_label,l=256):
    """
    随机从图片中取 变成为l的正方形
    :param img: 亮度矩阵
    :return:
    """
    #随机取一个点：
    cropimg_list = []
    start_y = random.randint(0,2160-l)
    start_x = random.randint(0,4096-l)
    for img in img_lumin_list:
        cropimg_list.append(img[start_y:start_y+l,start_x:start_x+l][np.newaxis, :]) #剪裁以后，添加维度。
    img_label_crop = img_label[start_y:start_y + l, start_x:start_x + l]
    # if mask is not
    # mask_crop = mask[start_y:start_y+l,start_x:start_x+l]

    return cropimg_list,img_label_crop
def Q99(img):
    """

    :param img: array 矩阵
    :return:
    """
    a,b = img.shape
    img_ = img.reshape(a*b,1)
    print(img_.shape)
def dellist(list_i,value):
    list_out = []
    for i in list_i:
        if i != 0:
            list_out.append(i)
    return list_out

def cal99_1(img):
    """
      :param img: array 矩阵
      :return:
      """
    a, b = img.shape
    # 将矩阵拉成条
    array_2 = img.reshape(1, a * b)[0, :]
    # 变成列表
    array_list = array_2.tolist()
    # 删除0
    array_list2 = dellist(array_list, 0)
    # 升序排列
    array_list_sorted = sorted(array_list2)  # 升序排列
    list_len = len(array_list_sorted)
    p1_idx = int(0.01 * list_len)
    p99_idx = int(0.99 * list_len)
    p1 = array_list_sorted[p1_idx]
    p99 = array_list_sorted[p99_idx]
    return p1,p99

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

        label1_path = self.index_pair[step].replace("intensity_norm","label").replace("A","label")

        imgA = np.load(img_pathA)
        imgC = np.load(img_pathC)
        imgG = np.load(img_pathG)
        imgT = np.load(img_pathT)
        label1 = np.load(label1_path)

        img_list = [imgA,imgC,imgG,imgT]
        #cropimg_list = []
        imgcrop_list,labelcrop = cropimage(img_list,label1)



        img = np.concatenate((imgcrop_list[0],imgcrop_list[1],imgcrop_list[2],imgcrop_list[3]))

        # 将img图中没call对的盖起来
        labelcrop2 = labelcrop.copy()
        labelcrop2[labelcrop2!=5] = 1
        labelcrop2[labelcrop2==5] = 0
        img = np.multiply(img,labelcrop2)

        #img1 = cv2.imread(img1_path,0)
        #label1 = cv2.imread(label1_path,0).astype('int64') #one hot 只能用LongTensor

        img_tensor = torch.from_numpy(img).float()
        labelcrop = labelcrop.astype('int64') #转化为长整形  。longtensor
        label1_tensor = torch.from_numpy(labelcrop)
        #img_tensor = img_tensor.unsqueeze(0)
        label1_tensor = torch.nn.functional.one_hot(label1_tensor,6) #(l,l,6)
        label1_tensor = label1_tensor[:,:,1:5].float() #不要最后没call 对的。0：背景  1-4 ：碱基  5： 没call对
        label1_tensor = label1_tensor.permute(2,0,1).contiguous()

        return img_tensor,label1_tensor



class Dataset_epoch_norm(data.Dataset):
    def __init__(self,names,norm=True):
        #输入只读带有A的
        super(Dataset_epoch_norm,self).__init__()
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

        label1_path = self.index_pair[step].replace("intensity","label").replace("A","label")

        imgA = np.load(img_pathA)
        imgC = np.load(img_pathC)
        imgG = np.load(img_pathG)
        imgT = np.load(img_pathT)
        label1 = np.load(label1_path)

        img_list = [imgA,imgC,imgG,imgT]
        #cropimg_list = []
        imgcrop_list,labelcrop = cropimage(img_list,label1)

        if self.norm == True:
            # 采用99分为点归一化
            for i in range(len(img_list)):
                p1,p99 = cal99_1(img_list[i])
                imgcrop_list[i] = imgnorm(imgcrop_list[i],p1,p99)

        #归一化，并切块的图
        #增加一个维度：
        img = np.concatenate((imgcrop_list[0],imgcrop_list[1],imgcrop_list[2],imgcrop_list[3]))

        # 将img图中没call对的盖起来
        labelcrop2 = labelcrop.copy()
        labelcrop2[labelcrop2!=5] = 1
        labelcrop2[labelcrop2==5] = 0
        img = np.multiply(img,labelcrop2)

        #img1 = cv2.imread(img1_path,0)
        #label1 = cv2.imread(label1_path,0).astype('int64') #one hot 只能用LongTensor

        img_tensor = torch.from_numpy(img).float()
        labelcrop = labelcrop.astype('int64') #转化为长整形  。longtensor
        label1_tensor = torch.from_numpy(labelcrop)
        #img_tensor = img_tensor.unsqueeze(0)
        label1_tensor = torch.nn.functional.one_hot(label1_tensor,6) #(l,l,6)
        label1_tensor = label1_tensor[:,:,:5].float() #不要最后没call 对的。0：背景  1-4 ：碱基  5： 没call对
        label1_tensor = label1_tensor.permute(2,0,1).contiguous()

        return img_tensor,label1_tensor


class Dataset_epoch_test(data.Dataset):
    def __init__(self,names,norm=True):
        #输入只读带有A的
        super(Dataset_epoch_test,self).__init__()
        self.names = names
        self.norm = norm
        #所有的数据都读到了index_pair里
        self.index_pair = names
    def __len__(self):
        return len(self.index_pair)

    def __getitem__(self, step):

        #把4个通道叠一起：

        img = np.random.rand(4,256,256)
        labelcrop = np.random.randint(0,5,(256,256))

        img_tensor = torch.from_numpy(img).float()
        labelcrop = labelcrop.astype('int64') #转化为长整形  。longtensor
        label1_tensor = torch.from_numpy(labelcrop)
        #img_tensor = img_tensor.unsqueeze(0)
        label1_tensor = torch.nn.functional.one_hot(label1_tensor,6) #(l,l,6)
        label1_tensor = label1_tensor[:,:,:5].float() #不要最后没call 对的。0：背景  1-4 ：碱基  5： 没call对
        label1_tensor = label1_tensor.permute(2,0,1).contiguous()

        return img_tensor,label1_tensor



