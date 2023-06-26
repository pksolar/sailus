import glob
import torch
import torch.utils.data as data
from data.datasets import Dataset_epoch_test
from callNet.model1 import  DNA_Sequencer
import math
import numpy as np
import json
import fastQ
import time


def main():
    dataset_name = "08"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCHSIZE = 1
    N_CLASS = 4
    rate = 1
    classes = ['N', 'A', 'C', 'G', 'T']  # n0,a1,c2,g3,t4
    test_dir_total = glob.glob("E:\code\python_PK\callbase\datasets\{}\Res\Lane01\deepLearnData\*\intensity_norm\R001C001_A.npy".format(dataset_name)) #total_dir = glob.glob(r"E:\code\python_PK\callbase\datasets\{}\Res\Lane01\deepLearnData\*\intensity_norm\*_A.npy".format(dataset_name))

    test_dir = test_dir_total
    test_dataset = Dataset_epoch_test(test_dir)
    test_loader = data.DataLoader(test_dataset, batch_size=BATCHSIZE, shuffle=False,num_workers=4)

    model = DNA_Sequencer().to(device)
    best_model = torch.load("savedir_pth/acc99.299826.pth.tar")['state_dict']
    model.load_state_dict(best_model)
    idx = 0
    len_point = 0
    patch_size1 = 2160
    patch_size2 = 4096
    row = math.ceil(2160/patch_size1)
    col = math.ceil(4096/patch_size2)
    listacgt=[]
    list_p_total =[]
    list_p_reads = []
    list_p_cyc = []
    dictacgt = {1:"A",2:"C",3:"G",4:"T"}
    def softmax(x):
        return np.exp(x) / np.sum(np.exp(x), axis=1)
    def croptensor(input, label, msk, r, c): #img: b c h w
        rowmin = r * patch_size1
        rowmax = (r+1) * patch_size1
        if rowmax > input.shape[2]:
            rowmax = input.shape[2]
        colmin  = c * patch_size2
        colmax = (c+1) *  patch_size2
        if colmax > input.shape[3]:
            colmax = input.shape[3]
        input_crop = input[:,:,rowmin:rowmax,colmin:colmax]
        label_crop = label[:, :, rowmin:rowmax, colmin:colmax]
        msk_crop   = msk[:, :, rowmin:rowmax, colmin:colmax]
        return  input_crop,label_crop,msk_crop
    list_array = []
    with torch.no_grad():
        for r in range(row):
            print("r:",r)
            for c in range(col):
                print("c:",c)
                idx = 0
                for inputs, labels, msk in test_loader:  #
                    print("path: ",test_dir[idx])
                    idx  = idx +1
                    model.eval()
                    #对inputs和labels进行剪裁。
                    inputs = inputs.to(device)
                    print("input_shape:",inputs.shape[2])
                    labels = labels.to(device)  # labels size;b,c,h,w
                    msk = msk.to(device)

                    inputs,labels,msk = croptensor(inputs,labels,msk,r,c)
                    start = time.time()
                    outputs = model(inputs)
                    end = time.time()
                    print("model use time:",end-start)

                    _, preds = torch.max(outputs,1)  # preds size [b,h,w],值是0，1,2,3，label是0,1,,,哪个通道的概率值最大，输出是,第几个通道的值最大。，b is batchsize,h,w is the predicted result.save it.
                    _, label_ = torch.max(labels, 1)  # label的值也是0,1,2,3，
                    # 把outputs和变形前的msk相乘，然后把结果拉成 b c h*w,再删除对应全为0的。得到b c reads_num ,把这样的结果叠加100个cycle.最后调整维度顺序。



                    issame = (preds == label_)  # 里面有多少true和false，
                    msk = torch.squeeze(abs(msk),1) #msk b,1,b,w
                    total = torch.sum(msk).item()
                    right = torch.sum(msk* issame).item()
                    accurate = 100 * right/total
                    print("total accuray: %.2f %%" % (accurate))
                    # 只统计abs(msk)为1的区域：
                    listacgt_idx = 0 #第几个reads

                    outputs_np = outputs.cpu().numpy()
                    outputs_np_softmax = softmax(outputs_np)

                    preds_np = preds.cpu().numpy().astype(np.int)
                    msk_np = msk.cpu().numpy().astype(np.int)
                    # msk_np = np.astype("int")
                    start = time.time()
                    for i in range(preds_np.shape[1]):
                        # print("第%d行"%i)
                        for j in range(preds_np.shape[2]):
                            pred = preds_np[0,i,j] + 1
                            if msk_np[0,i,j] == 1:

                                if idx == 1:  # 说明是第一个cycle，创建包含几个reads的列表 ，
                                    listacgt.append(dictacgt[pred])
                                    # list_p_cyc=outputs_np_softmax[0,:,i,j].tolist()
                                    # list_p_reads.append(list_p_cyc)
                                    # list_p_total.append(list_p_reads)
                                    # list_p_reads = []

                                    #listacgt[-1] = dictacgt[preds[i][j]]
                                #     pass
                                else: #第二个cycle以后依次写入：
                                    listacgt[listacgt_idx+len_point] = listacgt[listacgt_idx+len_point] + dictacgt[pred]
                                #     # list_p_total[listacgt_idx+len_point].append(outputs_np_softmax[0,:,i,j].tolist())
                                    listacgt_idx = listacgt_idx + 1
                                #     pass
                    end = time.time()
                    print("ergodic use time:",end-start)
                    print(idx)
                    # print(listacgt[:2])
                    # print(list_p_total[:2])
                len_point = len(listacgt)

    filename1 = 'ACGT08_2.json'
    with open(filename1, 'w') as file_obj:
        json.dump(listacgt, file_obj)

    # filename2 = 'possibility08.json'
    # with open(filename2, 'w') as file_obj2:
    #     json.dump(list_p_total, file_obj2)

    fastQ.writeFq('fastq/fast08_2.fq',listacgt,'ROO1C001')


                    # # 这个统计好好写。
                    # # 展平c和label
                    # c = c.flatten(0)  # 是不是不必拍平。
                    # label_ = label_.flatten(0)
                    # msk = msk.flatten(0)
                    # 只比有mask的位置，也只统计这里的位置。
                    # accurate = 100 * torch.sum(class_correct) / torch.sum(class_total)
                    # accurate_show.update(accurate.item())
if __name__=='__main__':
    main()

