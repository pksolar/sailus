import os

import torch

from datasets import Dataset_3cycle,Dataset_3cycle_val
from callNet import  DNA_Sequencer,Max
from  model_unet import UNet_lowDense as UNet
from utils import  *
import numpy as np
import json
import fastQ
import time
from datasets_read_img import Dataset_3cycle_test
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCHSIZE = 1
    fov = "144h_R001C001"
    test_dir_total =  glob.glob(
        rf"C:\deepdata\image\img\{fov}_*_A_img.tif") #total_dir = glob.glob(r"E:\code\python_PK\callbase\datasets\{}\Res\Lane01\deepLearnData\*\intensity_norm\*_A.npy".format(dataset_name))
    test_dir = test_dir_total
    test_dataset = Dataset_3cycle_test(test_dir)
    test_loader = data.DataLoader(test_dataset, batch_size=BATCHSIZE, shuffle=False,num_workers=1)

    model = UNet().to(device)
    # best_model = torch.load("full_Unet/acc98.8321.pth.tar")['state_dict']
    save_dir = 'savedir_pth/'
    model_lists = natsorted(glob.glob(save_dir + '*'))
    model_load = model_lists[-2]
    print(model_load)
    best_model = torch.load(model_load)['state_dict']

    model.load_state_dict(best_model)

    criterion = nn.CrossEntropyLoss()
    listacgt=[]
    dictacgt = {1:"A",2:"C",3:"G",4:"T"}

    accurate_show = AverageMeter()
    loss_val_show = AverageMeter()
    cycle = 1
    with torch.no_grad():
        model.eval()
        for inputs, labels, msk in test_loader:  #注意此处的msk 是否包含mapping上的

            inputs = inputs.to(device)
            labels = labels.to(device)  # labels size;b,c,h,w
            msk = msk.to(device)
            outputs = model(inputs)
            #loss_val = criterion(outputs * msk, labels * msk)
            #print("loss_val:", loss_val.item())
            #loss_val_show.update(loss_val.item())

            preds = torch.max(outputs,1)[1] * msk   # preds size [b,h,w],值是0，1,2,3，label是0,1,,,哪个通道的概率值最大，输出是,第几个通道的值最大。，b is batchsize,h,w is the predicted result.save it.
            label_ = torch.max(labels, 1)[1] * msk # label的值也是0,1,2,3，
            # 用mask把不必要的不去比较。mask是，0,1 变成，（-10,0）,然后和label以及preds相加。得到preds和label是负值的地方就是背景，拍平成列表，再删除也元素，得到很短的列表，再去比较。速度会大大加快。


            preds = torch.squeeze(preds).cpu().numpy().flatten().astype(int)
            msk = torch.squeeze(msk).cpu().numpy().flatten().astype(int)  # msk b,1,b,w
            label = torch.squeeze(label_).cpu().numpy().flatten().astype(int)

            idx = np.where(msk != 0)  # 只算mapping上的。
            preds = preds[idx]  #去除了背景 .transpose([2, 1, 0])  # numreads,1,channel  .squeeze().transpose() # numreads,channel
            label = label[idx]
            c = (preds == label)
            accurate = np.sum(c)/label.shape[0] * 100
            accurate_show.update(accurate)
            print(" accuray: %.3f %%" % (accurate))
            print("********toatal accurate:{:.3f}%********".format(accurate_show.avg))

            preds = preds + 1 # 从1开始。
            for i,pred in enumerate(preds):
                        if cycle == 1:  # 说明是第一个cycle，创建包含几个reads的列表 ，
                            listacgt.append(dictacgt[pred])
                        else: #第二个cycle以后依次写入：
                            listacgt[i] = listacgt[i] + dictacgt[pred]
            print("cycle:",cycle)
            cycle += 1
            if cycle == 20:
                break
            #print("ergodic use time:",end-start)
    timestr = time.strftime("%m%d%H%M%S")
    save_path = f'fastq/{fov}_{accurate_show.avg:.3f}_{timestr}_lowDense_98.11/Lane01//'
    os.makedirs(save_path,exist_ok=True)
    fastQ.writeFq(save_path+'Lane01_fastq.fq', listacgt, 'R001C001')
if __name__=='__main__':
    main()

