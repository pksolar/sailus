import glob

import torch
import torch.nn as nn
import cv2
import  numpy as np
import torch.utils.data as data
import os

class Onekernel(nn.Module):
    def __init__(self):
        super(Onekernel,self).__init__()
        self.conv1 = nn.Conv2d(1,1,3,padding = 1,bias=True)
    def forward(self,x):
        x = self.conv1(x)
        #print(self.conv1.weight)
        return  x,self.conv1.weight,self.conv1.bias

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        # print("self.avg:",self.avg)

def blockId(array,bId,num = 3):
    """
    :param array: 输入矩阵
    :param bId:
    :param block_number:
    :return:
    """
    block_num = num * num
    height = array.shape[-2]
    width = array.shape[-1]
    blocksize_h = int(height/3)
    blocksize_w = int(width/3)
    blockimg = array[:,blocksize_h:blocksize_h*2,]




class Dataset_npy(data.Dataset):
    def __init__(self, data_array,label_y):  # daa_array: pic_num,height,cycle,channel,  label_array: pic_num,height,cycle
        super(Dataset_npy, self).__init__()
        self.data_array_path = data_array
        self.label_array_path = label_y # 100,4,4

    def __len__(self):
        return len(self.data_array_path)

    def __getitem__(self, idx):  #
        #要去除未map上的，否则无法和label_vector对应起来。
        input_array1 =  cv2.imread(self.data_array_path[idx],0)[np.newaxis,:]  #1,h,w
        input_array2 = cv2.imread(self.data_array_path[idx].replace("_A","_C"),0)[np.newaxis,:]
        input_array3 = cv2.imread(self.data_array_path[idx].replace("_A","_G"),0)[np.newaxis,:]
        input_array4 = cv2.imread(self.data_array_path[idx].replace("_A","_T"),0)[np.newaxis,:]

        label1 = np.load(self.label_array_path[idx])
        label2 = np.load(self.label_array_path[idx].replace("_A","_C"))
        label3 = np.load(self.label_array_path[idx].replace("_A","_G"))
        label4 = np.load(self.label_array_path[idx].replace("_A","_T"))

        pathname_ = os.path.basename(self.label_array_path[idx]).split("_")[0]


        #区分块


        return  input_array1,input_array2,input_array3,input_array4,label1,label2,label3,label4,pathname_

def getSubpixelInten(source, input_pos,pad=1):
    """
    :param source:
    :param input_pos: 维度： 2 x  30w (点的个数)  第一个维度中0：x坐标，（h），1：y坐标 (w)
    :param scale:
    :param pad:
    :return:
    """

    sour_shape = source.shape
    (sh, sw) = (sour_shape[-2], sour_shape[-1])
    padding = pad * torch.ones((sour_shape[0], sour_shape[1], sh + 1, sw + 1)).to(device)
    padding[:, :, :-1, :-1] = source
    # 目标图像h,w

    # 生成grid,新图，存放 新图在老图上对应的坐标
    # 计算新图到老图上的坐标。

    # 拉平，这里和我的数据十分相似了。里面是对应的坐标
    x = input_pos[0]
    y = input_pos[1]

    # 计算取整的坐标，并拉平
    clip = torch.floor(input_pos)
    cx = clip[0]  # 整数化后的横坐标
    cy = clip[1]

    f1 = padding[:, :, cy.detach().cpu().numpy(), cx.detach().cpu().numpy()]
    f2 = padding[:, :, cy.detach().cpu().numpy() + 1, cx.detach().cpu().numpy()]
    f3 = padding[:, :, cy.detach().cpu().numpy(), cx.detach().cpu().numpy() + 1]
    f4 = padding[:, :, cy.detach().cpu().numpy() + 1, cx.detach().cpu().numpy() + 1]

    a = cx + 1 - x
    b = x - cx
    c = cy + 1 - y
    d = y - cy

    fx1 = a * f1 + b * f2
    fx2 = a * f3 + b * f4
    fy = c * fx1 + d * fx2

    return fy

# def read_corrd(corrd_path):


device = "cuda"
# 读取图像的路径
paths_x = glob.glob("E:\code\python_PK\callbase\datasets\highDens\Image\Lane01\*\R001C001_A.tif") #必须出去未map上的
#导入label数据
label_y = glob.glob(r"E:\code\python_PK\bleeding\gtvalue\label_vector\*_A_label_vector.npy")
#导入未map的点的id，去除未map的
noMap = set(np.loadtxt(r"E:\code\python_PK\callbase\datasets\highDens\result_outcorrd\Lane01\sfile\R001C001noMap.txt")) #npy文件。里面都是未map上的id，生成的output_vector要除去这些id

loss_show1 = AverageMeter()
loss_show2 = AverageMeter()
loss_show3 = AverageMeter()
loss_show4 = AverageMeter()


criterian1 = nn.MSELoss()
criterian2 = nn.MSELoss()
criterian3 = nn.MSELoss()
criterian4 = nn.MSELoss()


net1 = Onekernel().to(device)
net2 = Onekernel().to(device)
net3 = Onekernel().to(device)
net4 = Onekernel().to(device)
optimizer1 = torch.optim.Adam(net1.parameters(), lr=0.001)
optimizer2 = torch.optim.Adam(net2.parameters(), lr=0.001)
optimizer3 = torch.optim.Adam(net3.parameters(), lr=0.001)
optimizer4 = torch.optim.Adam(net4.parameters(), lr=0.001)

train_dataset = Dataset_npy(paths_x,label_y)
train_loader = data.DataLoader(train_dataset, batch_size=1, num_workers=0, shuffle=True, drop_last=True)

# #读取点的id和其对应的亚像素坐标：
# subloc = np.loadtxt(r"E:\code\python_PK\callbase\datasets\highDens\result_outcorrd\Lane01\sfile\nearReadSet\R001C001Outlier_Id.txt") # 56854,3(id,x,y)
# # realCall = np.loadtxt(r"")
# # num_points = subloc.shape[0]
#用来预先定义4个tensor，存放4个通道所有点的灰度值。
output_list_A = []
output_list_C = []
output_list_G = []
output_list_T = []


for i in range(1000):
    j = 0
    loss_show1.reset()
    loss_show2.reset()
    loss_show3.reset()
    loss_show4.reset()

    for xA,xC,xG,xT,yA,yC,yG,yT,name_num in train_loader:
        cycle_num = int(name_num[0])
        j  = j+1
        xA = xA.to(device).float()
        xC = xC.to(device).float()
        xG = xG.to(device).float()
        xT = xT.to(device).float()
        yA = yA.to(device).float()
        yC = yC.to(device).float()
        yG = yG.to(device).float()
        yT = yT.to(device).float()

        outputA, convA,bA = net1(xA) # batchsize,channel,height,width
        outputC, convC,bC = net2(xC)
        outputG, convG,bG = net3(xG)
        outputT, convT,bT = net4(xT)
        print("finish conv")

        #将output插值成亚像素精度的tensor，并放到tensor矩阵里。
        #读取第一个cycle，A通道的亚像素坐标。

        #读几个cycele的temp文件
        subloc_A = np.loadtxt(r"E:\code\python_PK\callbase\datasets\highDens\result_outcorrd\Lane01\sfile\R001C001_{}_A_otherCycChanCoord.txt".format(cycle_num),usecols = (2,3)).T
        subloc_C = np.loadtxt(
            r"E:\code\python_PK\callbase\datasets\highDens\result_outcorrd\Lane01\sfile\R001C001_{}_C_otherCycChanCoord.txt".format(
                cycle_num), usecols=(2, 3)).T

        subloc_G = np.loadtxt(
            r"E:\code\python_PK\callbase\datasets\highDens\result_outcorrd\Lane01\sfile\R001C001_{}_G_otherCycChanCoord.txt".format(
                cycle_num), usecols=(2, 3)).T

        subloc_T = np.loadtxt(
            r"E:\code\python_PK\callbase\datasets\highDens\result_outcorrd\Lane01\sfile\R001C001_{}_T_otherCycChanCoord.txt".format(
                cycle_num), usecols=(2, 3)).T
        # subloc_C = np.loadtxt(
        #     r"E:\code\python_PK\callbase\datasets\highDens\result_outcorrd\Lane01\sfile\R001C001_{}_C_my.temp".format(
        #         cycle_num)).tolist()
        # subloc_G = np.loadtxt(
        #     r"E:\code\python_PK\callbase\datasets\highDens\result_outcorrd\Lane01\sfile\R001C001_{}_G_my.temp".format(
        #         cycle_num)).tolist()
        # subloc_T = np.loadtxt(
        #     r"E:\code\python_PK\callbase\datasets\highDens\result_outcorrd\Lane01\sfile\R001C001_{}_T_my.temp".format(
        #         cycle_num)).tolist()
        print("11111111111111111")
        subloc_A = torch.from_numpy(subloc_A).to(device).float()
        output_tensor_A = getSubpixelInten(outputA,subloc_A)

        print("finish A chn interplote")
        # for id, [x, y] in enumerate(subloc_C):  # x,y是横纵坐标,id 是read Id
        #     # 剔除 no mapped  id:
        #     if id not in noMap:
        #         subvalue = getSubpixelInten(x, y, outputC[0, 0, :, :])  # output_vector_A[id] = subvalue
        #         output_list_C.append(subvalue)
        # print("finish C chn interplote")
        # for id, [x, y] in enumerate(subloc_G):  # x,y是横纵坐标,id 是read Id
        #     # 剔除 no mapped  id:
        #     if id not in noMap:
        #         subvalue = getSubpixelInten(x, y, outputG[0, 0, :, :])  # output_vector_A[id] = subvalue
        #         output_list_G.append(subvalue)
        # print("finish G chn interplote")
        # for id, [x, y] in enumerate(subloc_T):  # x,y是横纵坐标,id 是read Id
        #     # 剔除 no mapped  id:
        #     if id not in noMap:
        #         subvalue = getSubpixelInten(x, y, outputT[0, 0, :, :])  # output_vector_A[id] = subvalue
        #         output_list_T.append(subvalue)
        # print("finish T chn interplote")

        # output_tensor_C = torch.stack(output_list_C)
        # output_tensor_G = torch.stack(output_list_G)
        # output_tensor_T = torch.stack(output_list_T)

        lossA = criterian1(output_tensor_A,yA.unsqueeze(0))
        print("lossA finish")
        # lossC = criterian2(output_tensor_C, yC.squeeze(0))
        # lossG = criterian3(output_tensor_G, yG.squeeze(0))
        # lossT = criterian4(output_tensor_T, yT.squeeze(0))



        loss_show1.update((lossA.item()))
        # loss_show2.update((lossC.item()))
        # loss_show3.update((lossG.item()))
        # loss_show4.update((lossT.item()))


        optimizer1.zero_grad()
        # optimizer2.zero_grad()
        # optimizer3.zero_grad()
        # optimizer4.zero_grad()
        lossA.backward()
        # lossC.backward()
        # lossG.backward()
        # lossT.backward()

        optimizer1.step()
        # optimizer2.step()
        # optimizer3.step()
        # optimizer4.step()
        output_list_A = []
        output_list_C = []
        output_list_G = []
        output_list_T = []

        print("iter:", j, loss_show1.avg, loss_show2.avg, loss_show3.avg, loss_show4.avg)
        #print("A:", convA, " ->", bA, "\nC:", convC, " ->", bC, "\nG:", convG, " ->", bG, "\nT", convT, " ->", bT)
        print("A:",convA,"->",bA)
        # if j % 5 == 0:
        #     print("pic:{}, A:{:.5f}, C:{:.5f},  G:{:.5f},  T:{:.5f}".format(j, loss.item(), loss2.item(), loss3.item(),
        #                                                                 loss4.item()))
    print("-------------------------------------------------------------------------------------------------------------")
    print("epoch:",i,loss_show1.avg,loss_show2.avg,loss_show3.avg,loss_show4.avg)
    print("A:", convA, " ->", bA, "\nC:", convC, " ->", bC, "\nG:", convG, " ->", bG, "\nT", convT, " ->", bT)
    print(
        "*****************************************************************************************************************")



