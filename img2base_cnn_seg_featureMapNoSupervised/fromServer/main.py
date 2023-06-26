import glob
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.utils.data as data
import os
import time
from natsort import natsorted
# from datasets import Dataset_3cycle,Dataset_3cycle_val
from datasets_read_img_3label import Dataset_3cycle,Dataset_3cycle_val
from callNet import  DNA_Sequencer,Max
from utils import  *
from model_unet import UNet
import random
from collections import OrderedDict
def main():
    continuetrain = False
    BATCHSIZE = 16
    best_acc = 0
    save_dir = 'savedir_pth/'
    classes = ['A', 'C', 'G', 'T']  # n0,a1,c2,g3,t4
    dict_class = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_files = glob.glob(
        r"C:\deepdata\picture_new\img\*_A_img.tif")

    train_files = [f for f in all_files if "h_" in f and "144h_" not in f ]
    val_files = [f for f in all_files if "144h_"  in f]

    # train_dir = total_dir[0:int(rate * len(total_dir))]  #
    val_files = val_files[-11:-1]

    random.shuffle(train_files)

    train_dataset = Dataset_3cycle(train_files)
    train_loader = data.DataLoader(train_dataset, batch_size=BATCHSIZE, num_workers=10, shuffle=True,pin_memory=True)
    # val_dir = glob.glob(r"E:\code\python_PK\callbase\datasets\{}\Res\Lane01\deepLearnData\*\intensity_norm\*_A.npy".format(
    #     dataset_name_val))[:5]
    val_dataset = Dataset_3cycle_val(val_files)
    val_loader = data.DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = UNet().to(device)



    if continuetrain == True:
        save_dir = 'savedir_pth/'
        model_lists = natsorted(glob.glob(save_dir + '*'))
        print(model_lists[-3])
        best_model = torch.load(model_lists[-3])['state_dict']
        best_model = OrderedDict(list(best_model.items())[:-2])

        model.load_state_dict(best_model,strict=False)

        for k, v in model.named_parameters():
            if k in best_model.keys():
                v.requires_grad = False

        params = filter(lambda p: p.requires_grad, model.parameters())
        # 定义损失函数和优化器
    criterion = FocalLoss(gamma=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(5000):
        # 获取训练数据
        """Training"""
        loss_show = AverageMeter()
        idx = 0
        # random.shuffle(train_files)
        start = time.time()
        for inputs,labels,msk,name in train_loader:
            # print("name:",name)
            idx += 1
            model.train()
            inputs = inputs.to(device,non_blocking = True)
            labels = labels.to(device,non_blocking=True) #labels size;b,c,h,w
            msk = msk.to(device,non_blocking=True)

            # 前向传播
            outputs = model(inputs)
            # 计算损失
            #记录loss
            #3个focal loss
            labelsnp = labels.cpu().numpy()
            number =  torch.sum(msk)
            loss_before = criterion(outputs[:, 0:4,:,:]*msk,labels[:,0:4,:,:]*msk,number)
            loss_middle = criterion(outputs[:,4:8, :, :] * msk, labels[:, 4:8, :, :] * msk,number)
            loss_behind = criterion(outputs[:, 8:12, :, :] * msk, labels[:, 8:12, :, :] * msk,number)
            loss =(loss_before + loss_middle + loss_behind)/3.0
            # 记录loss
            _, preds = torch.max(outputs[:, 4:8, :, :],1)  # preds size [b,h,w],值是0，1,2,3，label是0,1,,,哪个通道的概率值最大，输出是,第几个通道的值最大。，b is batchsize,h,w is the predicted result.save it.
            _, label_ = torch.max(labels[:, 4:8, :, :], 1)  # label的值也是0,1,2,3，
            # 用mask把不必要的不去比较。mask是，0,1 变成，（-10,0）,然后和label以及preds相加。得到preds和label是负值的地方就是背景，拍平成列表，再删除也元素，得到很短的列表，再去比较。速度会大大加快。

            # c = (preds * msk == label_ * msk)  # 里面有多少true和false，  5 h w

            loss_show.update(loss.item())
            # 清空梯度
            optimizer.zero_grad()
            #反向传播
            loss.backward()
            #更新参数
            optimizer.step()
            #打印loss
            print('Iter{} of {} loss {:.6f}'.format(idx,len(train_loader),loss.detach().cpu().numpy().item()))
        end = time.time()
        print("epoch time: ",end-start)


        # 打印epoch的平均loss
        print("Epoch {} loss {:.6f}".format(epoch,loss_show.avg))
        """validaton"""
        if epoch % 2 == 0:
            accurate_show = AverageMeter()
            loss_val_show = AverageMeter()
            with torch.no_grad():
                for inputs, labels, msk in val_loader:  #
                    model.eval()
                    inputs = inputs.to(device)
                    labels = labels.to(device)  # labels size;b,c,h,w
                    msk = msk.to(device)
                    outputs = model(inputs)
                    number = torch.sum(msk)
                    loss_before = criterion(outputs[:, 0:4, :, :] * msk, labels[:, 0:4, :, :] * msk, number)
                    loss_middle = criterion(outputs[:, 4:8, :, :] * msk, labels[:, 4:8, :, :] * msk, number)
                    loss_behind = criterion(outputs[:, 8:12, :, :] * msk, labels[:, 8:12, :, :] * msk, number)
                    loss_val = (loss_before + loss_middle + loss_behind) / 3.0
                    print("loss_val:", loss_val.item())
                    loss_val_show.update(loss_val.item())
                    # 只统计msk的位置。
                    # 只比msk的位置。

                    msk = torch.squeeze(abs(msk))  # msk b,1,b,w

                    _, preds = torch.max(outputs[:,4:8,:,:],
                                         1)  # preds size [b,h,w],值是0，1,2,3，label是0,1,,,哪个通道的概率值最大，输出是,第几个通道的值最大。，b is batchsize,h,w is the predicted result.save it.
                    _, label_ = torch.max(labels[:,4:8,:,:], 1)  # label的值也是0,1,2,3，
                    # 用mask把不必要的不去比较。mask是，0,1 变成，（-10,0）,然后和label以及preds相加。得到preds和label是负值的地方就是背景，拍平成列表，再删除也元素，得到很短的列表，再去比较。速度会大大加快。

                    c = (preds * msk == label_ * msk)  # 里面有多少true和false，  5 h w

                    # msk = abs(msk)
                    total = torch.sum(msk).item()
                    right = torch.sum(msk * c).item()
                    accurate = 100 * right / total
                    accurate_show.update(accurate)
                    print(" accuray: %.2f %%" % (accurate))

            print("********Epoch:{},toatal accurate:{:.4f}%********".format(epoch, accurate_show.avg))

            best_acc = max(accurate_show.avg, best_acc)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }, save_dir=save_dir, filename='acc{:.4f}.pth.tar'.format(accurate_show.avg))



            loss_show.reset()
            accurate_show.reset()
            loss_val_show.reset()



if __name__ == '__main__':
    main()