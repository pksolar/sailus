import os
import glob
import numpy as np
import cv2
import torch.utils.data as Data

'''
通过继承Data.Dataset，实现将一组Tensor数据对封装成Tensor数据集
至少要重载__init__，__len__和__getitem__方法
'''

def normalize(x):
    Max = np.max(x)
    Min = np.min(x)
    x_ = (x-Min)/(Max-Min)
    return x_

class Dataset(Data.Dataset):
    def __init__(self, files_fixed,files_moving):
        # 初始化，files是所有文件，这里没有设置batch，batch就是所有 。
        self.files_fixed = files_fixed
        self.files_moving = files_moving

    def __len__(self):
        # 返回数据集的大小
        return len(self.files_moving)

    def __getitem__(self, index):
        # 索引数据集中的某个数据，还可以对数据进行预处理
        # 下标index参数是必须有的，名字任意
        #files确实是一个list，里面是所有数据的路径。
        #此处没有用csv文件做。其实也可以用csv文件做。


        img_arr_moving =cv2.imread(self.files_moving[index],0)[np.newaxis,:]

        type_ = self.files_moving[index].split("\\")[-1][0]

        fixedpath = self.files_moving[index].replace(type_,"A").replace("moving","fixed")


        img_arr_fixed = cv2.imread(fixedpath, 0)[np.newaxis, :]

        name = self.files_moving[index].split("\\")[-1]



        # 返回值自动转换为torch的tensor类型
        return normalize(img_arr_fixed),normalize(img_arr_moving),name


class DatasetTest(Data.Dataset):
    def __init__(self, files_fixed,files_moving):
        # 初始化，files是所有文件，这里没有设置batch，batch就是所有 。
        self.files_fixed = files_fixed
        self.files_moving = files_moving

    def __len__(self):
        # 返回数据集的大小
        return len(self.files_fixed)

    def __getitem__(self, index):
        # 索引数据集中的某个数据，还可以对数据进行预处理
        # 下标index参数是必须有的，名字任意
        #files确实是一个list，里面是所有数据的路径。
        #此处没有用csv文件做。其实也可以用csv文件做。


        img_arr_moving =cv2.imread(self.files_moving[index],0)[np.newaxis,:]

        type_ = self.files_moving[index].split("\\")[-1][0]

        fixedpath = self.files_moving[index].replace(type_,"A").replace("moving","fixed")


        img_arr_fixed = cv2.imread(fixedpath, 0)[np.newaxis, :]



        # 返回值自动转换为torch的tensor类型
        return normalize(img_arr_fixed),normalize(img_arr_moving)