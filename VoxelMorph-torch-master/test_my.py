import os
import glob
import numpy as np
import SimpleITK as sitk
import torch.utils.data as Data

'''
通过继承Data.Dataset，实现将一组Tensor数据对封装成Tensor数据集
至少要重载__init__，__len__和__getitem__方法
'''


class Dataset(Data.Dataset):
    def __init__(self, files):
        # 初始化
        self.files = files

    def __len__(self):
        # 返回数据集的大小
        return len(self.files)

    def __getitem__(self, index):
        # 索引数据集中的某个数据，还可以对数据进行预处理
        # 下标index参数是必须有的，名字任意
        img_arr = sitk.GetArrayFromImage(sitk.ReadImage(self.files[index]))[np.newaxis, ...]
        # 返回值自动转换为torch的tensor类型
        return img_arr


class Tag:
    def __init__(self, id):
        self.id = id

    def __getitem__(self, item):
        # print('这个方法被调用')
        return self.id


# -*- coding:utf-8 -*-
class DataTest:
    def __init__(self, id, address):
        self.id = id
        self.address = address
        self.d = {self.id: 1,
                  self.address: "192.168.1.1"
                  }
        self.list = range(100,320)
    def dog(self,idx):
        print("0: ",idx)
        return "cat"
    def __getitem__(self, key):
        print("aaa")
        return self.list[key]

    def __len__(self,idxx):
        print("32: ",len(self.list))


data = DataTest(1, "192.168.2.11")
print ("1: ",data.dog(90))
print("2: ",data[12])


