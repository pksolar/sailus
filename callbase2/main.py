import glob
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.utils.data as data
import cv2
import os
import numpy as np
from natsort import natsorted
from data.datasets import Dataset_epoch
from callNet.model1 import  DNA_Sequencer2
from utils.utils import AverageMeter


def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=8):
    torch.save(state, save_dir+filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))
def main():

    writer = SummaryWriter(log_dir='callNet_log')
    BATCHSIZE = 8
    N_CLASS = 5
    best_acc = 0
    save_dir = 'savedir_pth/'
    classes=['N','A','C','G','T'] #n0,a1,c2,g3,t4
    dict_class = {0: 'N', 1: 'A', 2: 'C', 3: 'G', 4: 'T'}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dir = glob.glob("data/image/train/*.jpg")
    train_dataset = Dataset_epoch(train_dir)
    train_loader = data.DataLoader(train_dataset,batch_size=BATCHSIZE,shuffle=True)

    val_dir = glob.glob("data/image/val/*.jpg")
    val_dataset = Dataset_epoch(val_dir)
    val_loader = data.DataLoader(val_dataset,batch_size=BATCHSIZE,shuffle=False)

    model = DNA_Sequencer().to(device)
    model2 = DNA_Sequencer2().to(device)


    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    for epoch in range(100):
        # 获取训练数据
        """Training"""
        loss_show = AverageMeter()
        idx = 0
        for inputs,labels in train_loader:
            idx += 1
            model.train()
            inputs = inputs.to(device)
            labels = labels.to(device) #labels size;b,c,h,w

            # 前向传播
            outputs = model(inputs)
            # 计算损失
            loss = criterion(outputs, labels)
            # 记录loss
            loss_show.update(loss.item())
            # 清空梯度
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            # 打印loss
            print('Iter{} of {} loss {:.4f}'.format(idx,len(train_loader),loss.detach().cpu().numpy().item()))

            # 处理最终输出
            outputs = model(inputs) # outputs:[b,5,h,w]
            loss = criterion(outputs,labels) # ce loss，github 上说还可以用 focal loss,weighted 等等

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        writer.add_scalar('Loss/train',loss_show.avg,epoch)
        # 打印epoch的平均loss
        print("Epoch {} loss {:.4f}",format(epoch,loss_show.avg))
        """validaton"""
        accurate_show = AverageMeter()

        accurate_showA = AverageMeter()
        accurate_showC = AverageMeter()
        accurate_showG = AverageMeter()
        accurate_showT = AverageMeter()
        accurate_showN = AverageMeter()

        accurate_show_list=[accurate_showN,
                            accurate_showA,
                            accurate_showC,
                            accurate_showG,
                            accurate_showT]


        loss_val_show = AverageMeter()

        with torch.no_grad():
            for inputs,labels in val_loader: #
                model.eval()

                inputs = inputs.to(device)
                labels = labels.to(device)  # labels size;b,c,h,w
                outputs = model(inputs)
                loss_val = criterion(outputs, labels)
                print("loss_val:",loss_val.item())
                loss_val_show.update(loss_val)
                #验证的精度:
                class_correct =  torch.zeros(N_CLASS)
                class_total =  torch.zeros(N_CLASS)
                _, preds = torch.max(outputs, 1) #preds size [b,h,w],哪个通道的概率值最大，输出是,第几个通道的值最大。，b is batchsize,h,w is the predicted result.save it.
                c = (preds == labels) #
                #展平c和label
                c = c.flatten(0)
                labels = labels.flatten(0)

                for i in range(c.shape[0]): #labels:  b c h w
                    label = labels[i] #label can only be 0-4
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

                for i in range(N_CLASS):
                    accurate_i = 100*class_correct[i]/class_total[i]
                    accurate_show_list[i].update(accurate_i)
                    print('Accuracy of %3s:%.2f %%'%(classes[i],100*class_correct[i]/class_total[i]))
                accurate =100 * torch.sum(class_correct) / torch.sum(class_total)
                accurate_show.update(accurate)
                print("total accuray: %.2f %%" % (accurate))

        print("Epoch:{},toatal accurate:{:.2f}".format(epoch,accurate_show.avg))
        writer.add_scalar('Val/accurate', accurate_show.avg, epoch)
        # 打印这个每个碱基的预测：
        for i in range(5):
            print(dict_class[i],": ",accurate_show_list[i].avg)
        writer.add_scalar('Val/NACGT_acc:',
                          {"N":accurate_showN.avg,
                           "A":accurate_showA.avg,
                           "C":accurate_showC.avg,
                           "G":accurate_showG.avg,
                           "T":accurate_showT.avg},
                            global_step = epoch)

        best_acc = max(accurate_show.avg,best_acc)
        save_checkpoint({
            'epoch':epoch + 1,
            'state_dict':model.state_dict(),
            'best_acc':best_acc,
            'optimizer':optimizer.state_dict(),
        },save_dir = save_dir,filename='dsc{:.3f}.pth.tar'.format(accurate.avg))

        #


        writer.add_scalar('Val/loss_val:',loss_val_show.avg,epoch)
        loss_show.reset()
        accurate_show.reset()
        loss_val_show.reset()

    writer.close()

if __name__=='__main__':
    main()
