import glob

import torch
import torch.nn as nn
import cv2
import  numpy as np
import torch.utils.data as data
import os


class weightConstraint(object):
    def __init__(self):
        pass

    def __call__(self, module):
        if hasattr(module, 'weight'):
            print("Entered")
            w11 = module.weight.data[:,:,1,1]
            w11 = w11.clamp(2, 8)  # 将参数范围限制到0.5-0.7之间

            w00 = module.weight.data[:,:,0,0]
            w01= module.weight.data[:, :, 0, 1]
            w02 = module.weight.data[:, :, 0, 2]
            w10 = module.weight.data[:, :, 1, 0]
            w12 = module.weight.data[:, :, 0, 0]
            w20 = module.weight.data[:, :, 2, 0]
            w21 = module.weight.data[:, :, 2, 1]
            w22 = module.weight.data[:, :, 2, 2]

            w00 = w00.clamp(-1,-0.5)
            w01 = w01.clamp(-1, 0)
            w02 = w02.clamp(-1, 0)
            w10 = w10.clamp(-1, 0)
            w12 = w12.clamp(-1, 0)
            w20 = w20.clamp(-1, 0)
            w21 = w21.clamp(-1, 0)
            w22 = w22.clamp(-1, 0)

            module.weight.data[:, :, 0, 0]=w00
            module.weight.data[:, :, 0, 1]=w01
            module.weight.data[:, :, 0, 2]=w02
            module.weight.data[:, :, 1, 0]=w10
            module.weight.data[:, :, 0, 0]=w12
            module.weight.data[:, :, 2, 0]=w20
            module.weight.data[:, :, 2, 1]=w21
            module.weight.data[:, :, 2, 2]=w22




class Onekernel(nn.Module):
    def __init__(self):
        super(Onekernel,self).__init__()
        self.conv1 = nn.Conv2d(1,1,3,padding = 1,bias=False)
        initial_weight_data = torch.Tensor([[[[-0.5,-0.75,-0.5],[-0.75,5.25,-0.75],[-0.5,-0.75,-0.5]]]])
        self.conv1.weight = nn.Parameter(initial_weight_data)
        print( self.conv1.weight.data)
    def forward(self,x):
        x = self.conv1(x)
        #print(self.conv1.weight)
        return  x,self.conv1.weight.data

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

        label1 = np.load(self.label_array_path[idx])[np.newaxis,:]

        pathname_ = os.path.basename(self.label_array_path[idx]).split("_")[0]


        #区分块


        return  input_array1,label1,pathname_

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
#导入未map的点的id，去除未map
loss_show = AverageMeter()

criterian = nn.MSELoss()

net = Onekernel().to(device)

constraints=weightConstraint()


net._modules['conv1'].apply(constraints)



optimizer = torch.optim.Adam(net.parameters(), lr=0.001)




train_dataset = Dataset_npy(paths_x,label_y)
train_loader = data.DataLoader(train_dataset, batch_size=1, num_workers=0, shuffle=True, drop_last=True)


for i in range(1000):
    j = 0
    loss_show.reset()

    for x,y,name_num in train_loader:

        for p in net.parameters():
            print("P：", p)
            p = p.data.clamp(-1, -0.5)

        cycle_num = int(name_num[0])
        j  = j+1
        x = x.to(device).float()
        y = y.to(device).float()

        output, conv= net(x) # batchsize,channel,height,width

        subloc = np.loadtxt( r"E:\code\python_PK\callbase\datasets\highDens\result_outcorrd\Lane01\sfile\R001C001_{}_A_otherCycChanCoord.txt".format(cycle_num), usecols=(2, 3)).T

        subloc = torch.from_numpy(subloc).to(device).float()

        output_tensor = getSubpixelInten(output,subloc)


        loss = criterian(output_tensor,y)




        loss_show.update((loss.item()))



        optimizer.zero_grad()


        loss.backward()


        optimizer.step()





        print("iter:", j,",loss:{:.6f}".format(loss_show.avg))
        print("A conv:",conv)
    print("-------------------------------------------------------------------------------------------------------------")
    print("iter:", j, ",loss:{:.6f}".format(loss_show.avg))
    print("A conv:", conv)
    print("*****************************************************************************************************************")



