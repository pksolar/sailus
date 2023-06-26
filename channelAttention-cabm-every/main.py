import torch
import glob
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.utils.data as data
from datasets_ import Dataset_npy,Dataset_npy_val
import os
from natsort import natsorted
import numpy as np
from mymodel import DNA_Sequencer_Atten

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
    save_dir = 'savedir_pth/'
    best_acc = 0
    batchsize = 8
    height = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_array = np.load("data/08_R001C001_conv.npy")[:224496] #224496
    label_array = np.load("data/08_R001C001_label_conv.npy")[:224496] #
    val_data = np.load("data/08_R002C034_conv.npy")[:10000]
    val_label = np.load("data/08_R002C034_label_conv.npy")[:10000]

    num,cycle_num,channel = data_array.shape
    num_pic = np.floor(num/height).astype(int)

    num2, cycle_num2, channel2 = val_data.shape
    num_pic2 = np.floor(num2 / height).astype(int)


    data_array = data_array[:num_pic * height].reshape(num_pic,height,cycle_num,channel)
    label_array = label_array[:num_pic * height].reshape(num_pic,height,cycle_num)
    train_dataset = Dataset_npy(data_array, label_array)
    train_loader = data.DataLoader(train_dataset, batch_size=batchsize, num_workers=4, shuffle=True, drop_last=True)

    val_data = val_data[:num_pic2 * height].reshape(num_pic2, height, cycle_num2, channel2)
    val_label = val_label[:num_pic2 * height].reshape(num_pic2, height, cycle_num2)
    val_dataset = Dataset_npy_val(val_data, val_label)
    val_loader = data.DataLoader(val_dataset, batch_size=batchsize, num_workers=0, shuffle=False, drop_last=True)

    model = DNA_Sequencer_Atten().to(device)
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10000):
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
            #print('Iter{} of {} loss {:.4f}'.format(idx, len(train_loader), loss.detach().cpu().numpy().item()))
            #设计一个卷积网络，带注意力

            _, pred = torch.max(outputs, 1)  # pred 取 0,1,2,3
            _, label = torch.max(labels, 1)  #
            c = (pred == label)  # 里面有多少true和false，
            right = torch.sum(c).item()
            accurate = 100 * right / (pred.shape[0] * pred.shape[1] * pred.shape[2])
            accurate_show.update(accurate)
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
                print('VAL [%d] loss = %.3f, acc:%.3f %%' % (epoch + 1, loss.item(), accurate))
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

if __name__=="__main__":
    main()