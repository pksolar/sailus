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
def cropimage(img_lumin_list,img_label,mask,l=256):
    """
    随机从图片中取 变成为l的正方形
    :param img: 亮度矩阵
    :return:
    """
    #随机取一个点：
    cropimg_list = []
    start_y = random.randint(0,2160-l)
    start_x = random.randint(0,4096-l)
    # print("start_y:{},start_x:{}".format(start_y,start_x))
    for img in img_lumin_list:
        cropimg_list.append(img[start_y:start_y+l,start_x:start_x+l][np.newaxis, :]) #剪裁以后，添加维度。
    img_label_crop = img_label[start_y:start_y + l, start_x:start_x + l]
    mask_crop = mask[start_y:start_y + l, start_x:start_x + l]
    # if mask is not
    # mask_crop = mask[start_y:start_y+l,start_x:start_x+l]

    return cropimg_list,img_label_crop,mask_crop
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

def cropimage_val(img_lumin_list,img_label,mask,l=256):
    """
    随机从图片中取 变成为l的正方形
    :param img: 亮度矩阵
    :return:
    """
    #随机取一个点：
    cropimg_list = []
    start_y = 1000
    start_x = 1800
    # print("start_y:{},start_x:{}".format(start_y,start_x))
    for img in img_lumin_list:
        cropimg_list.append(img[start_y:start_y+l,start_x:start_x+l][np.newaxis, :]) #剪裁以后，添加维度。
    img_label_crop = img_label[start_y:start_y + l, start_x:start_x + l]
    mask_crop = mask[start_y:start_y + l, start_x:start_x + l]
    # if mask is not
    # mask_crop = mask[start_y:start_y+l,start_x:start_x+l]

    return cropimg_list,img_label_crop,mask_crop

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


class Dataset_epoch12_val(data.Dataset):
    def __init__(self,names,norm=False):
        #输入只读带有A的
        super(Dataset_epoch12_val,self).__init__()
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
        dataset_name = img_pathA.split("\\")[5]
        cycle_name = img_pathA.split("\\")[9]


        cyc_num = int(cycle_name[-3:])
        if cycle_name != 'Cyc002':
            cyc0 = cyc_num - 1
            cyc0_name = 'Cyc' + str(cyc0).zfill(3)
            img_cyc0_pathA = img_pathA.replace(cycle_name,cyc0_name)
            img_cyc0_pathC = img_pathC.replace(cycle_name, cyc0_name)
            img_cyc0_pathG = img_pathG.replace(cycle_name, cyc0_name)
            img_cyc0_pathT = img_pathT.replace(cycle_name, cyc0_name)



            imgA_cyc0  = np.load(img_cyc0_pathA)
            imgC_cyc0 = np.load(img_cyc0_pathC)
            imgG_cyc0 = np.load(img_cyc0_pathG)
            imgT_cyc0 = np.load(img_cyc0_pathT)


        if cycle_name !='Cycl100':
            cyc2 = cyc_num + 1
            cyc2_name = 'Cyc'+str(cyc2).zfill(3)
            img_cyc2_pathA = img_pathA.replace(cycle_name, cyc2_name)
            img_cyc2_pathC = img_pathC.replace(cycle_name, cyc2_name)
            img_cyc2_pathG = img_pathG.replace(cycle_name, cyc2_name)
            img_cyc2_pathT = img_pathT.replace(cycle_name, cyc2_name)

            imgA_cyc2 = np.load(img_cyc2_pathA)
            imgC_cyc2 = np.load(img_cyc2_pathC)
            imgG_cyc2 = np.load(img_cyc2_pathG)
            imgT_cyc2 = np.load(img_cyc2_pathT)

        # 找到前一个cyc，和后一个cyc：

        #total_dir = glob.glob(r"E:\code\python_PK\callbase\datasets\08\Res\Lane01\deepLearnData\*\intensity_norm\*_A.npy")

        mask_path_dir =r"E:\code\python_PK\callbase\datasets\{}\Res\Lane01\deepLearnData\\".format(dataset_name)
        mask_path = mask_path_dir + img_pathA.split("\\")[-1].replace("A","mask")
        #print("mask_:",mask_path)
        #读取所有图片：
        mask = np.load(mask_path)
        imgA = np.load(img_pathA)
        imgC = np.load(img_pathC)
        imgG = np.load(img_pathG)
        imgT = np.load(img_pathT)
        label1 = np.load(label1_path)

        a, b = imgA.shape
        if cycle_name == 'Cyc002':
            # 如果是第一个cycle001,前面都补0
            imgA_cyc0 = np.zeros((a, b))
            imgG_cyc0 = np.zeros((a, b))
            imgC_cyc0 = np.zeros((a, b))
            imgT_cyc0 = np.zeros((a, b))

        if cycle_name == 'Cyc100':
            imgA_cyc2 = np.zeros((a, b))
            imgG_cyc2 = np.zeros((a, b))
            imgC_cyc2 = np.zeros((a, b))
            imgT_cyc2 = np.zeros((a, b))



        img_list = [imgA_cyc0,imgC_cyc0,imgG_cyc0,imgT_cyc0,imgA,imgC,imgG,imgT,imgA_cyc2,imgC_cyc2,imgG_cyc2,imgT_cyc2]

        #剪裁img，msk，label
        imgcrop_list,labelcrop,maskcrop = cropimage_val(img_list,label1,mask,512) #最后数字是剪裁大小

        #把msk的-1变成0：
        maskcrop[maskcrop==-1]=0 #1,256,256



        img = np.concatenate((imgcrop_list[0],imgcrop_list[1],imgcrop_list[2],imgcrop_list[3],imgcrop_list[4],
                              imgcrop_list[5],imgcrop_list[6],imgcrop_list[7],imgcrop_list[8],imgcrop_list[9],
                              imgcrop_list[10],imgcrop_list[11]))

        # 将img图中没call对的盖起来
        labelcrop2 = labelcrop.copy()
        labelcrop2[labelcrop2!=5] = 1
        labelcrop2[labelcrop2==5] = 0
        img = np.multiply(img,labelcrop2)

        #img1 = cv2.imread(img1_path,0)
        #label1 = cv2.imread(label1_path,0).astype('int64') #one hot 只能用LongTensor

        img_tensor = torch.from_numpy(img).float()
        maskcrop_tensor = torch.from_numpy(maskcrop).float()
        labelcrop = labelcrop.astype('int64') #转化为长整形  。longtensor
        label1_tensor = torch.from_numpy(labelcrop)
        maskcrop_tensor = maskcrop_tensor.unsqueeze(0)
        label1_tensor = torch.nn.functional.one_hot(label1_tensor,6) #(l,l,6)
        label1_tensor = label1_tensor[:,:,1:5].float() #不要最后没call 对的。0：背景  1-4 ：碱基  5： 没call对
        label1_tensor = label1_tensor.permute(2,0,1).contiguous()

        return img_tensor,label1_tensor,maskcrop_tensor



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
        dataset_name = img_pathA.split("\\")[5]
        cycle_name = img_pathA.split("\\")[9]


        cyc_num = int(cycle_name[-3:])
        if cycle_name != 'Cyc002':
            cyc0 = cyc_num - 1
            cyc0_name = 'Cyc' + str(cyc0).zfill(3)
            img_cyc0_pathA = img_pathA.replace(cycle_name,cyc0_name)
            img_cyc0_pathC = img_pathC.replace(cycle_name, cyc0_name)
            img_cyc0_pathG = img_pathG.replace(cycle_name, cyc0_name)
            img_cyc0_pathT = img_pathT.replace(cycle_name, cyc0_name)



            imgA_cyc0  = np.load(img_cyc0_pathA)
            imgC_cyc0 = np.load(img_cyc0_pathC)
            imgG_cyc0 = np.load(img_cyc0_pathG)
            imgT_cyc0 = np.load(img_cyc0_pathT)


        if cycle_name !='Cycl100':
            cyc2 = cyc_num + 1
            cyc2_name = 'Cyc'+str(cyc2).zfill(3)
            img_cyc2_pathA = img_pathA.replace(cycle_name, cyc2_name)
            img_cyc2_pathC = img_pathC.replace(cycle_name, cyc2_name)
            img_cyc2_pathG = img_pathG.replace(cycle_name, cyc2_name)
            img_cyc2_pathT = img_pathT.replace(cycle_name, cyc2_name)

            imgA_cyc2 = np.load(img_cyc2_pathA)
            imgC_cyc2 = np.load(img_cyc2_pathC)
            imgG_cyc2 = np.load(img_cyc2_pathG)
            imgT_cyc2 = np.load(img_cyc2_pathT)

        # 找到前一个cyc，和后一个cyc：

        #total_dir = glob.glob(r"E:\code\python_PK\callbase\datasets\08\Res\Lane01\deepLearnData\*\intensity_norm\*_A.npy")

        mask_path_dir =r"E:\code\python_PK\callbase\datasets\{}\Res\Lane01\deepLearnData\\".format(dataset_name)
        mask_path = mask_path_dir + img_pathA.split("\\")[-1].replace("A","mask")
        #print("mask_:",mask_path)
        #读取所有图片：
        mask = np.load(mask_path)
        imgA = np.load(img_pathA)
        imgC = np.load(img_pathC)
        imgG = np.load(img_pathG)
        imgT = np.load(img_pathT)
        label1 = np.load(label1_path)

        a, b = imgA.shape
        if cycle_name == 'Cyc002':
            # 如果是第一个cycle001,前面都补0
            imgA_cyc0 = np.zeros((a, b))
            imgG_cyc0 = np.zeros((a, b))
            imgC_cyc0 = np.zeros((a, b))
            imgT_cyc0 = np.zeros((a, b))

        if cycle_name == 'Cyc100':
            imgA_cyc2 = np.zeros((a, b))
            imgG_cyc2 = np.zeros((a, b))
            imgC_cyc2 = np.zeros((a, b))
            imgT_cyc2 = np.zeros((a, b))



        img_list = [imgA_cyc0,imgC_cyc0,imgG_cyc0,imgT_cyc0,imgA,imgC,imgG,imgT,imgA_cyc2,imgC_cyc2,imgG_cyc2,imgT_cyc2]

        #剪裁img，msk，label
        imgcrop_list,labelcrop,maskcrop = cropimage(img_list,label1,mask)
        maskcrop[maskcrop==-1]=0 #1,256,256



        img = np.concatenate((imgcrop_list[0],imgcrop_list[1],imgcrop_list[2],imgcrop_list[3],imgcrop_list[4],
                              imgcrop_list[5],imgcrop_list[6],imgcrop_list[7],imgcrop_list[8],imgcrop_list[9],
                              imgcrop_list[10],imgcrop_list[11]))

        # 将img图中没call对的盖起来
        labelcrop2 = labelcrop.copy()
        labelcrop2[labelcrop2!=5] = 1
        labelcrop2[labelcrop2==5] = 0
        img = np.multiply(img,labelcrop2)

        #img1 = cv2.imread(img1_path,0)
        #label1 = cv2.imread(label1_path,0).astype('int64') #one hot 只能用LongTensor

        img_tensor = torch.from_numpy(img).float()
        maskcrop_tensor = torch.from_numpy(maskcrop).float()
        labelcrop = labelcrop.astype('int64') #转化为长整形  。longtensor
        label1_tensor = torch.from_numpy(labelcrop)
        maskcrop_tensor = maskcrop_tensor.unsqueeze(0)
        label1_tensor = torch.nn.functional.one_hot(label1_tensor,6) #(l,l,6)
        label1_tensor = label1_tensor[:,:,1:5].float() #不要最后没call 对的。0：背景  1-4 ：碱基  5： 没call对
        label1_tensor = label1_tensor.permute(2,0,1).contiguous()

        return img_tensor,label1_tensor,maskcrop_tensor



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
    def __init__(self,names,norm=False):
        #输入只读带有A的
        super(Dataset_epoch_test,self).__init__()
        self.names = names
        self.norm = norm
        #所有的数据都读到了index_pair里
        self.index_pair = names
    def __len__(self):
        return len(self.index_pair)

    def __getitem__(self, step):

        # 把4个通道叠一起：
        img_pathA = self.index_pair[step]
        img_pathC = img_pathA.replace("A", "C")
        img_pathG = img_pathA.replace("A", "G")
        img_pathT = img_pathA.replace("A", "T")

        label1_path = self.index_pair[step].replace("intensity_norm", "label").replace("A", "label")
        dataset_name = img_pathA.split("\\")[5]
        cycle_name = img_pathA.split("\\")[9]

        cyc_num = int(cycle_name[-3:])
        if cycle_name != 'Cyc002':
            cyc0 = cyc_num - 1
            cyc0_name = 'Cyc' + str(cyc0).zfill(3)
            img_cyc0_pathA = img_pathA.replace(cycle_name, cyc0_name)
            img_cyc0_pathC = img_pathC.replace(cycle_name, cyc0_name)
            img_cyc0_pathG = img_pathG.replace(cycle_name, cyc0_name)
            img_cyc0_pathT = img_pathT.replace(cycle_name, cyc0_name)

            imgA_cyc0 = np.load(img_cyc0_pathA)
            imgC_cyc0 = np.load(img_cyc0_pathC)
            imgG_cyc0 = np.load(img_cyc0_pathG)
            imgT_cyc0 = np.load(img_cyc0_pathT)

        if cycle_name != 'Cycl100':
            cyc2 = cyc_num + 1
            cyc2_name = 'Cyc' + str(cyc2).zfill(3)
            img_cyc2_pathA = img_pathA.replace(cycle_name, cyc2_name)
            img_cyc2_pathC = img_pathC.replace(cycle_name, cyc2_name)
            img_cyc2_pathG = img_pathG.replace(cycle_name, cyc2_name)
            img_cyc2_pathT = img_pathT.replace(cycle_name, cyc2_name)

            imgA_cyc2 = np.load(img_cyc2_pathA)
            imgC_cyc2 = np.load(img_cyc2_pathC)
            imgG_cyc2 = np.load(img_cyc2_pathG)
            imgT_cyc2 = np.load(img_cyc2_pathT)

        # 找到前一个cyc，和后一个cyc：

        # total_dir = glob.glob(r"E:\code\python_PK\callbase\datasets\08\Res\Lane01\deepLearnData\*\intensity_norm\*_A.npy")

        mask_path_dir = r"E:\code\python_PK\callbase\datasets\{}\Res\Lane01\deepLearnData\\".format(dataset_name)
        mask_path = mask_path_dir + img_pathA.split("\\")[-1].replace("A", "mask")
        # print("mask_:",mask_path)
        # 读取所有图片：
        mask = np.load(mask_path)
        imgA = np.load(img_pathA)
        imgC = np.load(img_pathC)
        imgG = np.load(img_pathG)
        imgT = np.load(img_pathT)
        label1 = np.load(label1_path)

        a, b = imgA.shape
        if cycle_name == 'Cyc002':
            # 如果是第一个cycle001,前面都补0
            imgA_cyc0 = np.zeros((a, b))
            imgG_cyc0 = np.zeros((a, b))
            imgC_cyc0 = np.zeros((a, b))
            imgT_cyc0 = np.zeros((a, b))

        if cycle_name == 'Cyc100':
            imgA_cyc2 = np.zeros((a, b))
            imgG_cyc2 = np.zeros((a, b))
            imgC_cyc2 = np.zeros((a, b))
            imgT_cyc2 = np.zeros((a, b))

        img_list2 = [imgA_cyc0, imgC_cyc0, imgG_cyc0, imgT_cyc0, imgA, imgC, imgG, imgT, imgA_cyc2, imgC_cyc2, imgG_cyc2,
                    imgT_cyc2]
        img_list = []
        for img_ in img_list2:
            img_list.append(img_[np.newaxis,:])

        # 剪裁img，msk，label
        #test 不必剪裁
        #imgcrop_list, labelcrop, maskcrop = cropimage(img_list, label1, mask)
        mask = abs(mask)  # 1,256,256

        img = np.concatenate((img_list[0], img_list[1], img_list[2], img_list[3], img_list[4],
                              img_list[5], img_list[6], img_list[7], img_list[8], img_list[9],
                              img_list[10], img_list[11]))


        # 将img图中没call对的盖起来，没有call对的也不盖。
        # labelcrop2 = label1.copy()
        # labelcrop2[labelcrop2 != 5] = 1
        # labelcrop2[labelcrop2 == 5] = 0
        #img = np.multiply(img, labelcrop2)

        # img1 = cv2.imread(img1_path,0)
        # label1 = cv2.imread(label1_path,0).astype('int64') #one hot 只能用LongTensor

        img_tensor = torch.from_numpy(img).float()
        mask_tensor = torch.from_numpy(mask).float()
        label = label1.astype('int64')  # 转化为长整形  。longtensor
        label1_tensor = torch.from_numpy(label)
        mask_tensor = mask_tensor.unsqueeze(0)
        label1_tensor = torch.nn.functional.one_hot(label1_tensor, 6)  # (l,l,6)
        label1_tensor = label1_tensor[:, :, 1:5].float()  # 不要最后没call 对的。0：背景  1-4 ：碱基  5： 没call对
        label1_tensor = label1_tensor.permute(2, 0, 1).contiguous()

        return img_tensor, label1_tensor, mask_tensor



