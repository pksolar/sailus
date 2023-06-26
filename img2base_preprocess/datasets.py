import os, glob
import random

import torch, sys
from torch.utils.data import Dataset
# import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch.utils.data as data




class Dataset_3cycle(data.Dataset):
    """
     r"E:\data\deep\image2base\image\image\*.tif")z
    """
    def __init__(self,names,norm=False):
        #输入只读带有A的
        super(Dataset_3cycle,self).__init__()
        self.names = names
        #所有的数据都读到了index_pair里
        self.index_pair = names
    def __len__(self):
        return len(self.index_pair)
    def __getitem__(self, step):
        """
        直接读取npy数据，将前后cycle的拼接。如果时第一个cycle那就拼0

        这里的step是随机的。那step
        """

        path = self.index_pair[step] #'E:\\data\\deepData\\train\\label\\52.1h\\026\\093.npy'

        name_file = os.path.basename(path)
        name_list = path.split("\\")

        #img 换成label，026 找025 和027，找不到就取全0：
        machine = name_list[-3]
        cycname = name_list[-2]
        cyc_m_number = int(cycname)


        #middle的时候：
        img_middle = np.load(path)
        # np.save("debug/"+name_file,img_middle)
        c,w,d = img_middle.shape

        label_middle_path = path.replace("img", "label")  # basetype: A.tif
        # name_file = name_file + " label:" + os.path.basename(label_path)
        label_middle = np.load(label_middle_path)

        # front的时候：
        cyc_f_number = cyc_m_number - 1
        img_front_path = path.replace(machine+"\\"+cycname,rf"{machine}\\{cyc_f_number:03d}" )
        label_front_path = img_front_path.replace("img","label")

        try:
            img_front = np.load(img_front_path)
            label_front = np.load(label_front_path)
        except:
            img_front = np.zeros((4, w, d))
            label_front = np.zeros((1,w, d))



        # behind的时候：
        cyc_h_number = cyc_m_number + 1
        img_behind_path = path.replace(machine + "\\" + cycname, rf"{machine}\\{cyc_h_number:03d}")
        label_behind_path = img_behind_path.replace("img", "label")

        try:
            img_behind = np.load(img_behind_path)
            label_behind = np.load(label_behind_path)
        except:
            img_behind = np.zeros((4, w, d))
            label_behind = np.zeros((1,w, d))


        #concate the three cycle img
        img = np.concatenate([img_front,img_middle,img_behind])
        label = np.concatenate([label_front, label_middle, label_behind])


        # np.save("debug/label_" + name_file, label)
        # 这个msk必须时已经经过膨胀处理的mask。
        msk_path = path.replace("img","msk").replace(machine + "\\" + cycname,machine)
        msk = np.load(msk_path)
        #name_file = name_file + " msk:"+os.path.basename(msk_path)
        msk[msk<0] = 0
        # np.save("debug/msk_" + name_file , msk)


        # width = 256
        # height = 256
        # crop_size_x = random.randint(0, img.shape[1] - width)
        # crop_size_y = random.randint(0, img.shape[2] - height)
        # img = img[:, crop_size_x: crop_size_x + width, crop_size_y:crop_size_y + height].copy()
        # label = label[:,crop_size_x: crop_size_x + width, crop_size_y:crop_size_y + height].copy()
        # msk = msk[crop_size_x: crop_size_x + width, crop_size_y:crop_size_y + height].copy()


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
    """
     r"E:\data\deep\image2base\image\image\*.tif")z
    """
    def __init__(self,names,norm=False):
        #输入只读带有A的
        super(Dataset_3cycle_val,self).__init__()
        self.names = names
        #所有的数据都读到了index_pair里
        self.index_pair = names
    def __len__(self):
        return len(self.index_pair)
    def __getitem__(self, step):
        """
        直接读取npy数据，将前后cycle的拼接。如果时第一个cycle那就拼0

        这里的step是随机的。那step
        """
        path = self.index_pair[step] #'E:\\data\\deepData\\train\\label\\52.1h\\026\\093.npy'

        name_file = os.path.basename(path)
        name_list = path.split("\\")

        #img 换成label，026 找025 和027，找不到就取全0：
        machine = name_list[-3]
        cycname = name_list[-2]
        cyc_m_number = int(cycname)


        #middle的时候：
        img_middle = np.load(path)
        # np.save("debug/"+name_file,img_middle)
        c,w,d = img_middle.shape

        label_middle_path = path.replace("img", "label")  # basetype: A.tif
        # name_file = name_file + " label:" + os.path.basename(label_path)
        label_middle = np.load(label_middle_path)

        # front的时候：
        cyc_f_number = cyc_m_number - 1
        img_front_path = path.replace(machine+"\\"+cycname,rf"{machine}\\{cyc_f_number:03d}" )
        label_front_path = img_front_path.replace("img","label")

        try:
            img_front = np.load(img_front_path)
            label_front = np.load(label_front_path)
        except:
            img_front = np.zeros((4, w, d))
            label_front = np.zeros((1,w, d))



        # behind的时候：
        cyc_h_number = cyc_m_number + 1
        img_behind_path = path.replace(machine + "\\" + cycname, rf"{machine}\\{cyc_h_number:03d}")
        label_behind_path = img_behind_path.replace("img", "label")

        try:
            img_behind = np.load(img_behind_path)
            label_behind = np.load(label_behind_path)
        except:
            img_behind = np.zeros((4, w, d))
            label_behind = np.zeros((1,w, d))


        #concate the three cycle img
        img = np.concatenate([img_front,img_middle,img_behind])
        label = np.concatenate([label_front, label_middle, label_behind])


        # np.save("debug/label_" + name_file, label)
        # 这个msk必须时已经经过膨胀处理的mask。
        msk_path = path.replace("img","msk").replace(machine + "\\" + cycname,machine)
        msk = np.load(msk_path)
        #name_file = name_file + " msk:"+os.path.basename(msk_path)
        msk[msk<0] = 0
        # np.save("debug/msk_" + name_file , msk)

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





class Dataset_3cycle_val_mapping(data.Dataset):
    """
     r"E:\data\deep\image2base\image\image\*.tif")z
    """
    def __init__(self,names,norm=False):
        #输入只读带有A的
        super(Dataset_3cycle_val_mapping,self).__init__()
        self.names = names
        #所有的数据都读到了index_pair里
        self.index_pair = names
    def __len__(self):
        return len(self.index_pair)
    def __getitem__(self, step):
        """
        直接读取npy数据，将前后cycle的拼接。如果时第一个cycle那就拼0

        这里的step是随机的。那step
        """
        path = self.index_pair[step] #'E:\\data\\deepData\\train\\label\\52.1h\\026\\093.npy'

        name_file = os.path.basename(path)
        name_list = path.split("\\")

        #img 换成label，026 找025 和027，找不到就取全0：
        machine = name_list[-3]
        cycname = name_list[-2]
        cyc_m_number = int(cycname)


        #middle的时候：
        img_middle = np.load(path)
        # np.save("debug/"+name_file,img_middle)
        c,w,d = img_middle.shape

        label_middle_path = path.replace("img", "label")  # basetype: A.tif
        # name_file = name_file + " label:" + os.path.basename(label_path)
        label_middle = np.load(label_middle_path)

        # front的时候：
        cyc_f_number = cyc_m_number - 1
        img_front_path = path.replace(machine+"\\"+cycname,rf"{machine}\\{cyc_f_number:03d}" )
        label_front_path = img_front_path.replace("img","label")

        try:
            img_front = np.load(img_front_path)
            label_front = np.load(label_front_path)
        except:
            img_front = np.zeros((4, w, d))
            label_front = np.zeros((1,w, d))


        # behind的时候：
        cyc_h_number = cyc_m_number + 1
        img_behind_path = path.replace(machine + "\\" + cycname, rf"{machine}\\{cyc_h_number:03d}")
        label_behind_path = img_behind_path.replace("img", "label")

        try:
            img_behind = np.load(img_behind_path)
            label_behind = np.load(label_behind_path)
        except:
            img_behind = np.zeros((4, w, d))
            label_behind = np.zeros((1,w, d))


        #concate the three cycle img
        img = np.concatenate([img_front,img_middle,img_behind])
        label = np.concatenate([label_front, label_middle, label_behind])


        # np.save("debug/label_" + name_file, label)
        # 这个msk必须时已经经过膨胀处理的mask。
        msk_path = path.replace("img","msk").replace(machine + "\\" + cycname,machine)
        msk = np.load(msk_path)
        #name_file = name_file + " msk:"+os.path.basename(msk_path)
        msk = abs(msk)
        # np.save("debug/msk_" + name_file , msk)

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


class Dataset_3cycle_test(data.Dataset):
    """
     r"E:\data\deep\image2base\image\image\*.tif")z
    """
    def __init__(self,names,norm=False):
        #输入只读带有A的
        super(Dataset_3cycle_test,self).__init__()
        self.names = names
        #所有的数据都读到了index_pair里
        self.index_pair = names
    def __len__(self):
        return len(self.index_pair)
    def __getitem__(self, step):
        """
        直接读取npy数据，将前后cycle的拼接。如果时第一个cycle那就拼0

        这里的step是随机的。那step
        """
        path = self.index_pair[step] #'E:\\data\\deepData\\train\\label\\52.1h\\026\\093.npy'

        name_file = os.path.basename(path)
        name_list = path.split("\\")

        #img 换成label，026 找025 和027，找不到就取全0：
        machine = name_list[-3]
        cycname = name_list[-2]
        cyc_m_number = int(cycname)


        #middle的时候：
        img_middle = np.load(path)
        # np.save("debug/"+name_file,img_middle)
        c,w,d = img_middle.shape

        label_middle_path = path.replace("img", "label")  # basetype: A.tif
        # name_file = name_file + " label:" + os.path.basename(label_path)
        label_middle = np.load(label_middle_path)

        # front的时候：
        cyc_f_number = cyc_m_number - 1
        img_front_path = path.replace(machine+"\\"+cycname,rf"{machine}\\{cyc_f_number:03d}" )
        label_front_path = img_front_path.replace("img","label")

        try:
            img_front = np.load(img_front_path)
            label_front = np.load(label_front_path)
        except:
            img_front = np.zeros((4, w, d))
            label_front = np.zeros((1,w, d))



        # behind的时候：
        cyc_h_number = cyc_m_number + 1
        img_behind_path = path.replace(machine + "\\" + cycname, rf"{machine}\\{cyc_h_number:03d}")
        label_behind_path = img_behind_path.replace("img", "label")

        try:
            img_behind = np.load(img_behind_path)
            label_behind = np.load(label_behind_path)
        except:
            img_behind = np.zeros((4, w, d))
            label_behind = np.zeros((1,w, d))


        #concate the three cycle img
        img = np.concatenate([img_front,img_middle,img_behind])
        label = np.concatenate([label_front, label_middle, label_behind])


        # np.save("debug/label_" + name_file, label)
        # 这个msk必须时已经经过膨胀处理的mask。
        msk_path = path.replace("img","msk").replace(machine + "\\" + cycname,machine)
        msk = np.load(msk_path)
        #name_file = name_file + " msk:"+os.path.basename(msk_path)
        msk = abs(msk)
        # np.save("debug/msk_" + name_file , msk)

        pad_width_img = ((0, 0), (10, 10), (0, 0))
        pad_width = ((10, 10), (0, 0))
        img = np.pad(img, pad_width_img, mode='constant')
        label = np.pad(label, pad_width_img, mode='constant')
        msk = np.pad(msk, pad_width, mode='constant')



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
