import torch
import glob
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.utils.data as data
from datasets_ import Dataset_npy, Dataset_npy_val
import os
from natsort import natsorted
import numpy as np
from mymodel import DNA_Sequencer_Atten
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR
import time


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


def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=8):
    torch.save(state, save_dir + filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))


def npy_read_as_list(path1,path2,label=False,height=1000):
    data_array = np.load(path1)  # 224496
    label_array = np.load(path2)  #
    if label== True: #如果是测试集的label，没必要用所有reads，只用2w个即可。
        data_array = data_array[:20000]
        label_array = label_array[:20000]
    num, cycle_num, channel = data_array.shape
    num_pic = np.floor(num / height).astype(int)
    data_list = []
    data_label_list = []
    #将训练数据切成1000xcylex4的块。但是最后会余下不能区分的数据
    for ii in range(num_pic):
        data_list.append(data_array[height*ii:height*(ii+1)])
        data_label_list.append(label_array[height*ii:height*(ii+1)])
    #补齐最后的数据，可以用0补齐，也可以用从后往前切进行。此处选择从后往前切
    data_list.append(data_array[-height:])
    data_label_list.append(label_array[-height:])
    return  data_list,data_label_list

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        N, C,_,_ = inputs.size()
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            FL_loss = alpha_t * (1 - pt) ** self.gamma * BCE_loss
        else:
            FL_loss = (1 - pt) ** self.gamma * BCE_loss
        return FL_loss.mean()

def main():
    dataset_name = '08'

    dictacgt = {1: 'A', 2: "C", 3: "G", 4: "T"}
    rate = 1
    """
    输入的数据is npy : num x cycle x channel     25w x 100 x 4 
    把 25w 分成n个图, 1000 x  250  切成如此。
    #不会在外面设计数据。

    two direction


    """
    continueTrain = True
    save_dir = 'savedir_pth/'
    best_acc = 0
    batchsize = 4
    height = 1000
    dir_name = r"E:\data\liangdujuzhen\img"

    all_img_files = glob.glob(r"E:\data\liangdujuzhen\update_bingji\img\*.npy")
    train_img_files = [f for f in all_img_files if "08h"  in f and "pure" in f]
    train_label_files = [f.replace("img","label") for f in train_img_files]
    print(len(train_img_files), train_img_files)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_list = []
    data_label_list = []
    for path,path_label in zip(train_img_files,train_label_files):
        data_list_, data_label_list_ = npy_read_as_list(path, path_label)
        data_list = data_list + data_list_
        data_label_list = data_label_list + data_label_list_

    val_img_files = [f for f in all_img_files if "08h"  in f and "pure" in f]
    val_label_files = [f.replace("img", "label") for f in val_img_files]

    val_data_list,val_label_list = npy_read_as_list(val_img_files[0],val_label_files[0],label=True)


    # data_array = data_array[:num_pic * height].reshape(num_pic, height, cycle_num, channel)
    # label_array = label_array[:num_pic * height].reshape(num_pic, height, cycle_num)
    """
    用的是list来进行，而不是array
    """



    train_dataset = Dataset_npy(data_list, data_label_list)
    train_loader = data.DataLoader(train_dataset, batch_size=batchsize, num_workers=0, shuffle=True, drop_last=True)


    val_dataset = Dataset_npy_val(val_data_list, val_label_list)
    val_loader = data.DataLoader(val_dataset, batch_size=batchsize, num_workers=0, shuffle=False, drop_last=True)
    model = DNA_Sequencer_Atten().to(device)
    if continueTrain == True:
        best_model = torch.load("savedir_pth/acc99.603_epoch9932_loss0.0135.pth.tar")['state_dict']
        model.load_state_dict(best_model)
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss() #可以使用FocalLoss试试
    # criterion = FocalLoss(gamma=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    #scheduler = MultiStepLR(optimizer, milestones=[5000,10000], gamma=0.5,last_epoch=-1) #0.1  0.01 0.001 0.0001,0.00001
    lr=[]
    for epoch in range(30000):
        loss_show = AverageMeter()
        accurate_show = AverageMeter()
        idx = 0
        for inputs, labels in train_loader:
            idx += 1

            model.train()
            inputs = inputs.to(device)  # b,c,h,w
            labels = labels.to(device)  # b,c,h,w
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss_show.update(loss.item())
            # 清空梯度
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            #scheduler.step()
            # 打印loss

            # 设计一个卷积网络，带注意力

            _, pred = torch.max(outputs, 1)  # pred 取 0,1,2,3
            _, label = torch.max(labels, 1)  #
            c = (pred == label)  # 里面有多少true和false，
            right = torch.sum(c).item()
            accurate = 100 * right / (pred.shape[0] * pred.shape[1] * pred.shape[2])
            accurate_show.update(accurate)
            #print('[%d] loss = %.3f, acc:%.3f %%' % (epoch + 1, loss.item(), accurate))
            # print("total accuray:%.2f %%" % (accurate_show.avg))
        print('Epoch [%d] loss = %.3f, total acc: %.3f %%' % (epoch + 1, loss_show.avg, accurate_show.avg))
        # loss_show.reset()
        # accurate_show.reset()
        # with torch.no_grad():
        #     for inputs, labels in val_loader:
        #         idx += 1
        #         model.eval()
        #         inputs = inputs.to(device)  # b,c,h,w
        #         labels = labels.to(device)  # b,c,h,w
        #         outputs = model(inputs)
        #         loss = criterion(outputs, labels)
        #         loss_show.update(loss.item())
        #
        #         _, pred = torch.max(outputs, 1)  # pred 取 0,1,2,3
        #         _, label = torch.max(labels, 1)  #
        #         c = (pred == label)  # 里面有多少true和false，
        #         right = torch.sum(c).item()
        #         accurate = 100 * right / (pred.shape[0] * pred.shape[1] * pred.shape[2])
        #         accurate_show.update(accurate)
        #         #print('VAL [%d] loss = %.3f, acc:%.3f %%' % (epoch + 1, loss.item(), accurate))
        #     print('VAL:Epoch [%d] loss = %.3f, val total acc: %.3f %%' % (epoch + 1, loss_show.avg, accurate_show.avg))
        best_acc = max(accurate_show.avg, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, save_dir=save_dir, filename='acc{:.3f}_epoch{}_loss{:.4f}.pth.tar'.format(accurate_show.avg,epoch,loss_show.avg))
        loss_show.reset()
        accurate_show.reset()



if __name__ == "__main__":
    main()