import glob
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.utils.data as data
import os
from natsort import natsorted
from datasets import Dataset_3cycle,Dataset_3cycle_val
from callNet import  DNA_Sequencer,Max
from utils import  *

def main():
    continuetrain = False
    # dataset_name1 = '08'
    # dataset_name2 = '30'
    # dataset_name_val = '21'
    BATCHSIZE = 8
    N_CLASS = 4
    best_acc = 0
    rate = 0.85
    val_num = 2
    save_dir = 'savedir_pth/'
    classes = ['A', 'C', 'G', 'T']  # n0,a1,c2,g3,t4
    dict_class = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    total_dir1 = glob.glob(
        r"E:\code\python_PK\tools\CNNlabel_img2base\img2base\21_imgdata_full\*.npy")
    # total_dir2 = glob.glob(
    #     r"E:\code\python_PK\callbase\datasets\{}\Res\Lane01\deepLearnData\*\intensity_norm\*_A.npy".format(dataset_name2))

    total_dir = total_dir1  # + total_dir2

    #
    train_dir = total_dir[0:int(rate * len(total_dir))]  #
    val_dir = total_dir[-11:-1]


    train_dataset = Dataset_3cycle(train_dir)
    train_loader = data.DataLoader(train_dataset, batch_size=BATCHSIZE, num_workers=8, shuffle=True)
    # val_dir = glob.glob(r"E:\code\python_PK\callbase\datasets\{}\Res\Lane01\deepLearnData\*\intensity_norm\*_A.npy".format(
    #     dataset_name_val))[:5]
    val_dataset = Dataset_3cycle_val(val_dir)
    val_loader = data.DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = DNA_Sequencer().to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    if continuetrain == True:
        model_data = torch.load("savedir_pth/acc99.278817.pth.tar")['state_dict']
        model.load_state_dict(model_data)

        print("model load success")



    for epoch in range(500):
        # 获取训练数据
        """Training"""
        loss_show = AverageMeter()
        idx = 0
        for inputs,labels,msk in train_loader:
            idx += 1
            model.train()
            inputs = inputs.to(device)
            labels = labels.to(device) #labels size;b,c,h,w
            msk = msk.to(device)

            # 前向传播
            outputs = model(inputs)
            # 计算损失
            loss = criterion(outputs*msk, labels*msk) # ce loss，github 上说还可以用 focal loss,weighted 等等
            # 记录loss
            loss_show.update(loss.item())
            # 清空梯度
            optimizer.zero_grad()
            #反向传播
            loss.backward()
            #更新参数
            optimizer.step()
            # 打印loss
            print('Iter{} of {} loss {:.6f}'.format(idx,len(train_loader),loss.detach().cpu().numpy().item()))



        # 打印epoch的平均loss
        print("Epoch {} loss {:.6f}".format(epoch,loss_show.avg))
        """validaton"""
        if (epoch+1) % 10 == 0:
            accurate_show = AverageMeter()
            loss_val_show = AverageMeter()
            with torch.no_grad():
                for inputs, labels, msk in val_loader:  #
                    model.eval()
                    inputs = inputs.to(device)
                    labels = labels.to(device)  # labels size;b,c,h,w
                    msk = msk.to(device)
                    outputs = model(inputs)
                    loss_val = criterion(outputs * msk, labels * msk)
                    print("loss_val:", loss_val.item())
                    loss_val_show.update(loss_val.item())
                    # 只统计msk的位置。
                    # 只比msk的位置。

                    msk = torch.squeeze(abs(msk))  # msk b,1,b,w

                    _, preds = torch.max(outputs,
                                         1)  # preds size [b,h,w],值是0，1,2,3，label是0,1,,,哪个通道的概率值最大，输出是,第几个通道的值最大。，b is batchsize,h,w is the predicted result.save it.
                    _, label_ = torch.max(labels, 1)  # label的值也是0,1,2,3，
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

            #

            loss_show.reset()
            accurate_show.reset()
            loss_val_show.reset()

if __name__ == '__main__':
    main()