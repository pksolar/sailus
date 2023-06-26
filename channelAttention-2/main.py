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
    if label== True:
        data_array = data_array[:20000]
        label_array = label_array[:20000]
    num, cycle_num, channel = data_array.shape
    num_pic = np.floor(num / height).astype(int)
    data_list = []
    data_label_list = []
    for ii in range(num_pic):
        data_list.append(data_array[height*ii:height*(ii+1)])
        data_label_list.append(label_array[height*ii:height*(ii+1)])
    return  data_list,data_label_list



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
    save_dir = 'savedir_pth/'
    best_acc = 0
    batchsize = 8
    height = 1000
    dir_name = r"E:\code\python_PK\tools\miji\flatten\pure\\"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path_total = glob.glob(r"E:\code\python_PK\tools\miji\flatten\pure\*.npy")
    path_list = []
    path_label_list = []
    for path in path_total:
        if "label" not in path: #this is a  label
            path_list.append(path)

        else: # this is data
            path_label_list.append(path)
    # path_list = [dir_name+"08_R001C001_pure.npy",
    #              dir_name+"30_R001C001_pure.npy",
    #              dir_name+"30_R001C022_pure.npy",
    #              dir_name+"08_R003C067_pure.npy"
    #
    #              ]
    # path_label_list = [dir_name+"08_R001C001_label_pure.npy",
    #                    dir_name+"30_R001C001_label_pure.npy",
    #                     dir_name+"30_R001C022_label_pure.npy",
    #                     dir_name+"08_R003C067_label_pure.npy"
    #                    ]
    data_list = []
    data_label_list = []
    for path,path_label in zip(path_list,path_label_list):
        data_list_, data_label_list_ = npy_read_as_list(path, path_label)
        data_list = data_list + data_list_
        data_label_list = data_label_list + data_label_list_

    dir_name_val = r"E:\code\python_PK\tools\miji\flatten\val\pure\\"
    val_path = dir_name_val+"21_R001C001_pure.npy"
    val_label_path = dir_name_val+"21_R001C001_label_pure.npy"

    # data_list,data_label_list = npy_read_as_list(path,path_label)
    val_data_list,val_label_list = npy_read_as_list(val_path,val_label_path,label=True)



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
    best_model = torch.load("savedir_pth2/acc99.583.pth.tar")['state_dict']
    model.load_state_dict(best_model)
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    #scheduler = MultiStepLR(optimizer, milestones=[5000,10000], gamma=0.5,last_epoch=-1) #0.1  0.01 0.001 0.0001,0.00001


    for epoch in range(10000):
        # lr.append(scheduler.get_last_lr()[0])
        # print(epoch, scheduler.get_last_lr()[0])

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
            # scheduler.step()
            # 打印loss
            # print('Iter{} of {} loss {:.4f}'.format(idx, len(train_loader), loss.detach().cpu().numpy().item()))
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
        loss_show.reset()
        accurate_show.reset()
        with torch.no_grad():
            for inputs, labels in val_loader:
                idx += 1
                model.eval()
                inputs = inputs.to(device)  # b,c,h,w
                labels = labels.to(device)  # b,c,h,w
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss_show.update(loss.item())

                _, pred = torch.max(outputs, 1)  # pred 取 0,1,2,3
                _, label = torch.max(labels, 1)  #
                c = (pred == label)  # 里面有多少true和false，
                right = torch.sum(c).item()
                accurate = 100 * right / (pred.shape[0] * pred.shape[1] * pred.shape[2])
                accurate_show.update(accurate)
                #print('VAL [%d] loss = %.3f, acc:%.3f %%' % (epoch + 1, loss.item(), accurate))
            print('VAL:Epoch [%d] loss = %.3f, val total acc: %.3f %%' % (epoch + 1, loss_show.avg, accurate_show.avg))
        best_acc = max(accurate_show.avg, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, save_dir=save_dir, filename='acc{:.3f}.pth.tar'.format(accurate_show.avg))
        loss_show.reset()
        accurate_show.reset()



if __name__ == "__main__":
    main()