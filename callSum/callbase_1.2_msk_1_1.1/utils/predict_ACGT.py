import glob
import torch
import torch.utils.data as data
from data.datasets import Dataset_epoch_test
from callNet.model1 import  DNA_Sequencer
import math




dataset_name = "21"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCHSIZE = 1
N_CLASS = 4
rate = 1
classes = ['N', 'A', 'C', 'G', 'T']  # n0,a1,c2,g3,t4
test_dir_total = glob.glob("E:\code\python_PK\callbase\datasets\{}\Res\Lane01\deepLearnData\*\intensity_norm\R001C001_A.npy".format(dataset_name)) #total_dir = glob.glob(r"E:\code\python_PK\callbase\datasets\{}\Res\Lane01\deepLearnData\*\intensity_norm\*_A.npy".format(dataset_name))

test_dir = test_dir_total[int(rate*(len(test_dir_total))):]
test_dataset = Dataset_epoch_test(test_dir)
test_loader = data.DataLoader(test_dataset, batch_size=BATCHSIZE, shuffle=False)

model = DNA_Sequencer().to(device)
best_model = torch.load("savedir_pth/acc99.672577.pth.tar")['state_dict']
model.load_state_dict(best_model)
idx = 0
patch_size = 512
row = math.floor(2160/patch_size)
col = math.floor(4096/patch_size)


def croptensor(input, label, msk, r, c): #img: b c h w
    rowmin = r * patch_size
    rowmax = (r+1) * patch_size
    if rowmax > input.shape[2]:
        rowmax = input.shape[2]
    colmin  = c * patch_size
    colmax = (r+1) * input.shape[3]
    if colmax > input.shape[3]:
        colmax = input.shape[3]
    input_crop = input[:,:,rowmin:rowmax,colmin:colmax]
    label_crop = label[:, :, rowmin:rowmax, colmin:colmax]
    msk_crop   = msk[:, :, rowmin:rowmax, colmin:colmax]
    return  input_crop,label_crop,msk_crop

with torch.no_grad():
    for i in range(row):
        for j in range(col):
            for inputs, labels, msk in test_loader:  #
                print("path: ",test_dir[idx])
                idx  = idx +1
                model.eval()
                #对inputs和labels进行剪裁。
                inputs = inputs.to(device)
                labels = labels.to(device)  # labels size;b,c,h,w
                msk = msk.to(device)

                inputs,labels,msk = croptensor(inputs,labels,msk,i,j)
                outputs = model(inputs)

                _, preds = torch.max(outputs,1)  # preds size [b,h,w],值是0，1,2,3，label是0,1,,,哪个通道的概率值最大，输出是,第几个通道的值最大。，b is batchsize,h,w is the predicted result.save it.
                _, label_ = torch.max(labels, 1)  # label的值也是0,1,2,3，
                # 用mask把不必要的不去比较。mask是，0,1 变成，（-10,0）,然后和label以及preds相加。得到preds和label是负值的地方就是背景，拍平成列表，再删除也元素，得到很短的列表，再去比较。速度会大大加快。

                c = (preds == label_)  # 里面有多少true和false，

                msk = abs(msk)
                total = torch.sum(msk).item()
                right = torch.sum(msk* c).item()
                accurate = 100 * right/total
                print("total accuray: %.2f %%" % (accurate))
                # # 这个统计好好写。
                # # 展平c和label
                # c = c.flatten(0)  # 是不是不必拍平。
                # label_ = label_.flatten(0)
                # msk = msk.flatten(0)
                # 只比有mask的位置，也只统计这里的位置。
                # accurate = 100 * torch.sum(class_correct) / torch.sum(class_total)
                # accurate_show.update(accurate.item())


