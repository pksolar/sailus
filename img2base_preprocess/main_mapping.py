import os
import time
# from datasets import Dataset_3cycle,Dataset_3cycle_val
from datasets import Dataset_3cycle, Dataset_3cycle_val
from utils import *
import sys
from model.model_NestedUnet import NestedUNet
from val_mapping import automap_delelte
import  threading

from config import *

def sort_key(string):
    # 提取字符串中的数字部分作为排序键
    return int(string.split("\\")[-1][:3])




def get_all_files(directory):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    return file_paths

def main():
    continuetrain = True
    BATCHSIZE = 16
    best_acc = 0
    save_dir = 'pth/'
    os.makedirs(save_dir, exist_ok=True)
    classes = ['A', 'C', 'G', 'T']  # n0,a1,c2,g3,t4
    dict_class = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_root_dir = r"E:\data\deepData\train\img"
    val_root_dir = r"E:\data\deepData\val\img"

    #dataset = ImageFolder(train_root_dir)
    # 获取文件夹下所有文件的路径
    train_files = get_all_files(train_root_dir)
    train_files = sorted(train_files, key=sort_key)

    val_files = get_all_files(val_root_dir)
    val_files = sorted(val_files, key=sort_key)[10800:11000]

    print(len(train_files))

    train_dataset = Dataset_3cycle(train_files)
    train_loader = data.DataLoader(train_dataset, batch_size=BATCHSIZE, num_workers=8, shuffle=True, pin_memory=True)
    # val_dir = glob.glob(r"E:\code\python_PK\callbase\datasets\{}\Res\Lane01\deepLearnData\*\intensity_norm\*_A.npy".format(
    #     dataset_name_val))[:5]
    val_dataset = Dataset_3cycle_val(val_files)
    val_loader = data.DataLoader(val_dataset, batch_size=BATCHSIZE,num_workers=4, shuffle=False)

    model = NestedUNet(12).to(device)

    if continuetrain == True:
        pth_dir = 'pth/'
        model_lists = natsorted(glob.glob(pth_dir + '*'))
        model_load = model_lists[-12]
        print(model_load)
        best_model = torch.load(model_load)['state_dict']
        model.load_state_dict(best_model)

    criterion = FocalLoss(gamma=2)
    # criterion = nn.CrossEntropyLoss() #可以使用FocalLoss试试
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(5000):
        # 获取训练数据
        """Training"""
        loss_show = AverageMeter()
        idx = 0
        # random.shuffle(train_files)
        start = time.time()
        for inputs, labels, msk, name in train_loader:
            # print("name:",name)
            idx += 1
            model.train()
            inputs = inputs.to(device, non_blocking=True)
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)  # labels size;b,c,h,w
            msk = msk.to(device, non_blocking=True)

            # 前向传播
            outputs = model(inputs)
            number = torch.sum(msk)
            # loss = criterion(outputs*msk, labels*msk,number) # ce loss，github 上说还可以用 focal loss,weighted 等等

            loss_before = criterion(outputs[:, 0:4, :, :] * msk, labels[:, 0:4, :, :] * msk, number)
            loss_middle = criterion(outputs[:, 4:8, :, :] * msk, labels[:, 4:8, :, :] * msk, number)
            loss_behind = criterion(outputs[:, 8:12, :, :] * msk, labels[:, 8:12, :, :] * msk, number)

            # loss_before = criterion(outputs[:, 0:4, :, :] * msk, labels[:, 0:4, :, :] * msk)*65536/number
            # loss_middle = criterion(outputs[:, 4:8, :, :] * msk, labels[:, 4:8, :, :] * msk)*65536/number
            # loss_behind = criterion(outputs[:, 8:12, :, :] * msk, labels[:, 8:12, :, :] * msk)*65536/number

            loss = (loss_before + 2 * loss_middle + loss_behind) / 3.0
            # 记录loss
            # 用mask把不必要的不去比较。mask是，0,1 变成，（-10,0）,然后和label以及preds相加。得到preds和label是负值的地方就是背景，拍平成列表，再删除也元素，得到很短的列表，再去比较。速度会大大加快。

            """
            一方面让图更加像gausss生成的图。如果不是gauss生成的图。

            """
            # 记录loss
            loss_show.update(loss.item())
            # 清空梯度
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            # 打印loss
            sys.stdout.write("\r" + 'Iter{} of {} loss {:.6f}'.format(idx, len(train_loader), loss.detach().cpu().numpy().item()))
            sys.stdout.flush()
            # print('Iter{} of {} loss {:.6f}'.format(idx, len(train_loader), loss.detach().cpu().numpy().item()))

            if idx % 500 == 0:
                #直接保存pth，往后运行
                #保存模型到模型缓冲区，开新线程计算这个模型的mapping rate，比较mapping rate，保存较高的mappnig rate的模型。
                #best_acc = max(accurate_show.avg, best_acc)
                print("val start:")
                os.makedirs("pth_buffer",exist_ok=True)
                pth_name ="pth_buffer/" + 'idx{}_epoch{}.pth.tar'.format(idx, epoch)
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                    },  pth_name)
                #拿这个保存的pth来进行多线程mapping，
                #多线程的函数里要有，对pth_buffer里的进行读取，然后进行mapping，得到maprete，改变全局变量，然后通过maprate来确定是否保存。
                mapping_thread = threading.Thread(target=automap_delelte,args=(pth_name,idx,epoch,))

                mapping_thread.start()

        end = time.time()
        print("epoch time: ", end - start)

        # 打印epoch的平均loss
        print("Epoch {} loss {:.6f}".format(epoch, loss_show.avg))
        #保存loss_avg的值：
        with open(rf"exp/loss.txt",'a') as f:
                str_ = str(epoch)+" "+str(loss_show.avg)+"\n"
                f.write(str_)
        loss_show.reset()



if __name__ == '__main__':
    main()