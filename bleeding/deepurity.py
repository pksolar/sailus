import glob

import torch
import torch.nn as nn
import cv2
import  numpy as np
import torch.utils.data as data

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

class Dataset_npy(data.Dataset):
    def __init__(self, data_array,label_array):  # daa_array: pic_num,height,cycle,channel,  label_array: pic_num,height,cycle
        super(Dataset_npy, self).__init__()
        self.data_array_path = data_array
        self.lable_array_path = label_array



    def __len__(self):
        return len(self.data_array_path)

    def __getitem__(self, idx):  #
        input_array = np.load(self.data_array_path[idx])[np.newaxis,:]  # height,cyc,cha
        input_array2 = np.load(self.data_array_path[idx].replace("_A","_C"))[np.newaxis,:]
        input_array3 = np.load(self.data_array_path[idx].replace("_A","_G"))[np.newaxis,:]
        input_array4 = np.load(self.data_array_path[idx].replace("_A","_T"))[np.newaxis,:]

        input_label = np.load(self.lable_array_path[idx])  # height,cyc
        input_label = np.transpose(input_label, [2, 0, 1])
        # make array to tensor


        return  input_array,input_array2,input_array3,input_array4, input_label

device = "cuda"
paths_x = glob.glob("E:\code\python_PK\callbase\datasets\highDens\Result\Lane01\deepLearnData_ori\*\intensity\R001C001_A.npy")
paths_y = glob.glob("gtvalue/gtimg_ori/*.npy")

msk = np.load("E:\code\python_PK\callbase\datasets\highDens\Result\Lane01\deepLearnData_final\R001C001_mask.npy").astype(int)
msk[msk < 1] = 0
mskt = msk[np.newaxis,np.newaxis,:]
mskt = torch.from_numpy(mskt).to(device).float()

loss_show = AverageMeter()
loss_show2 = AverageMeter()
loss_show3 = AverageMeter()
loss_show4 = AverageMeter()

xi = cv2.imread("method1/ori.jpg",0)[np.newaxis,np.newaxis,:,:]
yi = cv2.imread("method1/result.jpg",0)[np.newaxis,np.newaxis,:,:]

x = torch.tensor(xi).float()
y = torch.tensor(yi).float()
criterian = nn.MSELoss()
criterian2 = nn.MSELoss()
criterian3 = nn.MSELoss()
criterian4 = nn.MSELoss()


net = Onekernel().to(device)
net2 = Onekernel().to(device)
net3 = Onekernel().to(device)
net4 = Onekernel().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
optimizer2 = torch.optim.Adam(net2.parameters(), lr=0.001)
optimizer3 = torch.optim.Adam(net3.parameters(), lr=0.001)
optimizer4 = torch.optim.Adam(net4.parameters(), lr=0.001)

train_dataset = Dataset_npy(paths_x, paths_y)
train_loader = data.DataLoader(train_dataset, batch_size=8, num_workers=0, shuffle=True, drop_last=True)

for i in range(1000):
    j = 0
    loss_show.reset()
    loss_show2.reset()
    loss_show3.reset()
    loss_show4.reset()

    for x,x2,x3,x4,y in train_loader:
        j  = j+1
        x = x.to(device).float()
        x2 = x2.to(device).float()
        x3 = x3.to(device).float()
        x4 = x4.to(device).float()
        y = y.to(device).float()# 8,4,2160,4096
        output,conv,b = net(x)
        output2, conv2,b2 = net2(x) #8,1,2160,4096
        output3, conv3,b3 = net3(x)
        output4, conv4,b4 = net4(x)

        output_ = torch.cat([output*mskt,output2*mskt,output3*mskt,output4],dim=1)
        output_sum = torch.sum(output_,dim=1)

        output_sumy = torch.sum(y,dim=1)

        purity_Ax = output*mskt/(output_sum+0.000001)
        purity_Cx = output2*mskt / (output_sum+0.000001)
        purity_Gx = output3*mskt / (output_sum+0.000001)
        purity_Tx = output4*mskt / (output_sum+0.000001)

        purity_Ay = output_sumy[0] / (output_sum + 0.000001)
        purity_Cy = output_sumy[1] / (output_sum + 0.000001)
        purity_Gy = output_sumy[2] / (output_sum + 0.000001)
        purity_Ty = output_sumy[3] / (output_sum + 0.000001)






        #purity calculate:
        x_purity =



        loss = criterian(output*mskt,y)
        loss2 = criterian2(output2*mskt, y[:,1:2,:,:])
        loss3 = criterian3(output3*mskt, y[:,2:3,:,:])
        loss4 = criterian2(output4*mskt, y[:,3:4,:,:])


        loss_show.update((loss.item()))
        loss_show2.update((loss2.item()))
        loss_show3.update((loss3.item()))
        loss_show4.update((loss4.item()))


        optimizer.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        optimizer4.zero_grad()
        loss.backward()
        loss2.backward()
        loss3.backward()
        loss4.backward()
        optimizer.step()
        optimizer2.step()
        optimizer3.step()
        optimizer4.step()
        print("iter:",j)




        # if j % 5 == 0:
        #     print("pic:{}, A:{:.5f}, C:{:.5f},  G:{:.5f},  T:{:.5f}".format(j, loss.item(), loss2.item(), loss3.item(),
        #                                                                 loss4.item()))

    print("epoch:",i,loss_show.avg,loss_show2.avg,loss_show3.avg,loss_show4.avg)
    print("A:", conv, " ->", b, "\nC:", conv2, " ->", b2, "\nG:", conv3, " ->", b3, "\nT", conv4, " ->", b4)






import numpy as np


