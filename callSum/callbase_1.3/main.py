import glob
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.utils.data as data
import os
from natsort import natsorted
from data.datasets import Dataset_epoch_norm , Dataset_epoch_test,Dataset_epoch
from callNet.model1 import  DNA_Sequencer,Max
"""
    出现的问题：
    1、不同通道归一化后，背景灰度值被放到不同的亮度。比如0：0.127, 1:0.143
    


"""
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

    writer = SummaryWriter()
    BATCHSIZE = 4
    N_CLASS = 4
    best_acc = 0
    rate = 0.4
    val_num = 2
    save_dir = 'savedir_pth/'
    classes=['A','C','G','T'] #n0,a1,c2,g3,t4
    dict_class = { 0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_dir = glob.glob("E:\code\python_PK\callbase\datasets/08/Res/Lane01/deepLearnData/*/intensity_norm/*_A.npy")

    train_dir = total_dir[0:int(rate*len(total_dir))] #

    train_dataset = Dataset_epoch(train_dir)
    train_loader = data.DataLoader(train_dataset,batch_size=BATCHSIZE,shuffle=True)
    val_dir = total_dir[int(rate * len(total_dir)):int(rate * len(total_dir))+val_num] #
    val_dataset = Dataset_epoch(val_dir)
    val_loader = data.DataLoader(val_dataset,batch_size=BATCHSIZE,shuffle=False)

    model = DNA_Sequencer().to(device)


    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
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
            print('Iter{} of {} loss {:.4f}'.format(idx,len(train_loader),loss.detach().cpu().numpy().item()))


        writer.add_scalar('Loss/train',loss_show.avg,epoch)
        # 打印epoch的平均loss
        print("Epoch {} loss {:.4f}".format(epoch,loss_show.avg))
        """validaton"""
        accurate_show = AverageMeter()

        accurate_showA = AverageMeter()
        accurate_showC = AverageMeter()
        accurate_showG = AverageMeter()
        accurate_showT = AverageMeter()


        accurate_show_list=[
                            accurate_showA,
                            accurate_showC,
                            accurate_showG,
                            accurate_showT]


        loss_val_show = AverageMeter()

        with torch.no_grad():
            for inputs,labels,msk in val_loader: #
                model.eval()
                inputs = inputs.to(device)
                labels = labels.to(device)  # labels size;b,c,h,w
                msk = msk.to(device)
                outputs = model(inputs)
                loss_val = criterion(outputs*msk, labels*msk)
                print("loss_val:",loss_val.item())
                loss_val_show.update(loss_val.item())
                #只统计msk的位置。

                class_correct =  torch.zeros(N_CLASS)
                class_total =  torch.zeros(N_CLASS)
                #只比msk的位置。
                _, preds = torch.max(outputs, 1) # preds size [b,h,w],值是0，1,2,3，label是0,1,,,哪个通道的概率值最大，输出是,第几个通道的值最大。，b is batchsize,h,w is the predicted result.save it.
                _,label_ = torch.max(labels,1)# label的值也是0,1,2,3，
                # 用mask把不必要的不去比较。mask是，0,1 变成，（-10,0）,然后和label以及preds相加。得到preds和label是负值的地方就是背景，拍平成列表，再删除也元素，得到很短的列表，再去比较。速度会大大加快。

                c = (preds == label_) #里面有多少true和false，
                #这个统计好好写。
                #展平c和label
                c = c.flatten(0) #是不是不必拍平。
                label_ = label_.flatten(0)
                msk = msk.flatten(0)
                #只比有mask的位置，也只统计这里的位置。

                for i in range(c.shape[0]): #labels:  b c h w
                    if msk[i] == 1:  # 只统计有亮点的区域。不统计背景
                        label = label_[i] #label can only be 0-3
                        class_correct[label] += c[i].item() #
                        class_total[label] += 1 #
                print("class_correct/batch:", class_correct / BATCHSIZE)
                print("class_total/batch:",class_total/BATCHSIZE)
                for i in range(N_CLASS):
                    accurate_i = 100*class_correct[i]/class_total[i]
                    accurate_show_list[i].update(accurate_i)
                    print('Accuracy of %3s:%.2f %%'%(classes[i],100*class_correct[i]/class_total[i]))
                accurate =100 * torch.sum(class_correct) / torch.sum(class_total)
                accurate_show.update(accurate.item())
                print("total accuray: %.2f %%" % (accurate))

        print("Epoch:{},toatal accurate:{:.2f}".format(epoch,accurate_show.avg))
        writer.add_scalar('Val/accurate', accurate_show.avg, epoch)
        writer.add_scalars('Val/NACGT_acc',
                          {
                           "A":accurate_showA.avg,
                           "C":accurate_showC.avg,
                           "G":accurate_showG.avg,
                           "T":accurate_showT.avg
                          },
                            global_step = epoch)

        best_acc = max(accurate_show.avg,best_acc)
        # save_checkpoint({
        #     'epoch':epoch + 1,
        #     'state_dict':model.state_dict(),
        #     'best_acc':best_acc,
        #     'optimizer':optimizer.state_dict(),
        # },save_dir = save_dir,filename='dsc{:.3f}.pth.tar'.format(accurate_show.avg))

        #


        writer.add_scalar('Val/loss_val:',loss_val_show.avg,epoch)
        loss_show.reset()
        accurate_show.reset()
        loss_val_show.reset()

    writer.close()

if __name__=='__main__':
    main()
