import torch
import glob
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.utils.data as data
from datasets_ import Dataset_npy
import os
from natsort import natsorted
import numpy as np
from mymodel import DNA_Sequencer

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
        #print("self.avg:",self.avg)

def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=8):
    torch.save(state, save_dir+filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))


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
    batchsize = 8
    height = 500
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_array = np.load("08_R001C001_pure.npy")[:224496]
    label_array = np.load("08_R001C001_label_pure.npy")[:224496]
    num,cycle_num,channel = data_array.shape
    num_pic = np.floor(num/height).astype(int)

    data_array = data_array[:num_pic * height].reshape(num_pic,height,cycle_num,channel)
    label_array = label_array[:num_pic * height].reshape(num_pic,height,cycle_num)
    train_dataset = Dataset_npy(data_array, label_array)
    train_loader = data.DataLoader(train_dataset, batch_size=batchsize, num_workers=0, shuffle=True, drop_last=True)

    model = DNA_Sequencer().to(device)
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1000):
        loss_show = AverageMeter()
        accurate_show = AverageMeter()
        idx = 0
        for inputs,labels in train_loader:
            idx += 1
            model.train()
            inputs = inputs.to(device) # b,c,h,w
            labels = labels.to(device) # b,c,h,w
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss_show.update(loss.item())
            # 清空梯度
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            # 打印loss
            print('Iter{} of {} loss {:.4f}'.format(idx, len(train_loader), loss.detach().cpu().numpy().item()))
            #设计一个卷积网络，带注意力
            _, pred = torch.max(outputs, 1)  # pred 取 0,1,2,3
            _, label = torch.max(labels, 1)  #
            c = (pred == label)  # 里面有多少true和false，
            right = torch.sum(c).item()
            accurate = 100 * right / (pred.shape[0] * pred.shape[1] * pred.shape[2])
            accurate_show.update(accurate)
            print('[%d/15] loss = %.3f, acc:%.2f %%' % (epoch + 1, loss.item(), accurate))
            # print("total accuray:%.2f %%" % (accurate_show.avg))
        print('Epoch [%d/15] loss = %.3f, total acc: %.3f' % (epoch + 1, loss_show.avg, accurate_show.avg))
        loss_show.reset()
        accurate_show.reset()




if __name__=="__main__":
    main()