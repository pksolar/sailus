import time
# from datasets import Dataset_3cycle,Dataset_3cycle_val
from datasets import Dataset_3cycle, Dataset_3cycle_val
from utils import *

from model.model_NestedUnet import NestedUNet




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
    train_files = get_all_files(train_root_dir)#[:100]
    train_files = sorted(train_files, key=sort_key)

    val_files = get_all_files(val_root_dir)
    val_files = sorted(val_files, key=sort_key)[11900:11000]

    print(len(train_files))

    train_dataset = Dataset_3cycle(train_files)
    train_loader = data.DataLoader(train_dataset, batch_size=BATCHSIZE, num_workers=BATCHSIZE, shuffle=True, pin_memory=True)
    # val_dir = glob.glob(r"E:\code\python_PK\callbase\datasets\{}\Res\Lane01\deepLearnData\*\intensity_norm\*_A.npy".format(
    #     dataset_name_val))[:5]
    val_dataset = Dataset_3cycle_val(val_files)
    val_loader = data.DataLoader(val_dataset, batch_size=BATCHSIZE, shuffle=False)

    model = NestedUNet(12).to(device)

    if continuetrain == True:
        pth_dir = 'pth/'
        model_lists = natsorted(glob.glob(pth_dir + '*'))
        print(model_lists[-1])
        best_model = torch.load(model_lists[-1])['state_dict']
        model.load_state_dict(best_model)

    criterion = FocalLoss(gamma=2)
    # criterion = nn.CrossEntropyLoss() #可以使用FocalLoss试试
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    model.train()
    for idx in range(150000000):
        idx += 1
        # 获取训练数据
        """Training"""
        loss_show = AverageMeter()
        inputs, labels, msk,name = iter(train_loader).next()
        # random.shuffle(train_files)
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)  # labels size;b,c,h,w
        msk = msk.to(device, non_blocking=True)
        # 前向传播
        outputs = model(inputs)
        number = torch.sum(msk)

        loss_before = criterion(outputs[:, 0:4, :, :] * msk, labels[:, 0:4, :, :] * msk, number)
        loss_middle = criterion(outputs[:, 4:8, :, :] * msk, labels[:, 4:8, :, :] * msk, number)
        loss_behind = criterion(outputs[:, 8:12, :, :] * msk, labels[:, 8:12, :, :] * msk, number)
        loss = (loss_before + 2 * loss_middle + loss_behind) / 3.0

        # 记录loss
        loss_show.update(loss.item())
        # 清空梯度
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        # 打印loss
        print('Iter{} of {} loss {:.6f}'.format(idx, len(train_loader), loss.detach().cpu().numpy().item()))

        # 打印epoch的平均loss
        """validaton"""
        if idx % 50 == 0:
            accurate_show = AverageMeter()
            loss_val_show = AverageMeter()
            with torch.no_grad():
                for inputs, labels, msk,name in val_loader:  #
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

                    _, preds = torch.max(outputs[:, 4:8, :, :], 1)  # preds size [b,h,w],值是0，1,2,3，label是0,1,,,哪个通道的概率值最大，输出是,第几个通道的值最大。，b is batchsize,h,w is the predicted result.save it.
                    _, label_ = torch.max(labels[:, 4:8, :, :], 1)  # label的值也是0,1,2,3，
                    # 用mask把不必要的不去比较。mask是，0,1 变成，（-10,0）,然后和label以及preds相加。得到preds和label是负值的地方就是背景，拍平成列表，再删除也元素，得到很短的列表，再去比较。速度会大大加快。
                    c = (preds * msk == label_ * msk)  # 里面有多少true和false，  5 h w
                    # msk = abs(msk)
                    total = torch.sum(msk).item()
                    right = torch.sum(msk * c).item()
                    accurate = 100 * right / total
                    accurate_show.update(accurate)
                    print(" accuray: %.2f %%" % (accurate))

            print("********Epoch:{},toatal accurate:{:.4f}%********".format(idx, accurate_show.avg))

            best_acc = max(accurate_show.avg, best_acc)
            save_checkpoint({
                'epoch': idx,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }, save_dir=save_dir, filename='acc{:.4f}_epoch{}.pth.tar'.format(accurate_show.avg, idx))

            #

            loss_show.reset()
            accurate_show.reset()
            loss_val_show.reset()
            model.train()


if __name__ == '__main__':
    main()