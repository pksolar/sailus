import glob
import torch
import torch.utils.data as data
from data.datasets import Dataset_epoch_test
from callNet.model1 import  DNA_Sequencer
import math
import numpy as np
import json
import fastQ
import  time
import threading
"""
  3 label  only 只对中间一个做推理
  32 线程


"""
names_threading = globals()
def callbase(preds_np,msk_np,idx,listacgt,dictacgt,len_point): #传入的list 也必须是以copy 的形式传。

    return listacgt.copy() #每次做完一个cycle都合并到最终的list中去。





def main():
    dataset_name = "21"
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
    best_model = torch.load("savedir_pth/acc99.523198.pth.tar")['state_dict']
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
                    outputs = model(inputs)

                    _, preds0 = torch.max(outputs[:, 0:4, :, :], 1)  #
                    _, preds1 = torch.max(outputs[:, 4:8, :, :], 1)  #
                    _, preds2 = torch.max(outputs[:, 8:12, :, :], 1)  #

                    _, label0 = torch.max(labels[:, 0:4, :, :], 1)  # label的值也是0,1,2,3，
                    _, label1 = torch.max(labels[:, 4:8, :, :], 1)  # label的值也是0,1,2,3，
                    _, label2 = torch.max(labels[:, 8:12, :, :], 1)  # label的值也是0,1,2,3，

                    c0 = (preds0 == label0)  # 里面有多少true和false，  5 h w
                    c1 = (preds1 == label1)
                    c2 = (preds2 == label2)

                    msk = torch.squeeze(abs(msk))  # msk b,1,b,w
                    # msk = abs(msk)
                    total = torch.sum(msk).item() * 3
                    right0 = torch.sum(msk * c0).item()
                    right1 = torch.sum(msk * c1).item()
                    right2 = torch.sum(msk * c2).item()
                    accurate_total = 100 * (right0 + right1 + right2) / total
                    accurate0 = 100 * (right0 * 3) / total
                    accurate1 = 100 * (right1 * 3) / total
                    accurate2 = 100 * (right2 * 3) / total
                    print("accuray0: %.2f %%" % (accurate0))
                    print("accuray1: %.2f %%" % (accurate1))
                    print("accuray2: %.2f %%" % (accurate2))
                    print("total accuray: %.2f %%" % (accurate_total))
                    # 只统计abs(msk)为1的区域：
                    listacgt_idx = 0 #第几个reads

                    #outputs_np = outputs.cpu().numpy()
                    #outputs_np_softmax = softmax(outputs_np)

                    preds_np0 = preds0.cpu().numpy().astype(int)
                    preds_np1 = preds1.cpu().numpy().astype(int)
                    preds_np2 = preds2.cpu().numpy().astype(int)

                    msk_np = msk.cpu().numpy().astype(int)
                    #比较两种预测的准确度：
                    #1、 3个3个预测
                    # 2、一个一个预测
                    for i in range(preds_np1.shape[1]):
                        # print("第%d行"%i)
                        for j in range(preds_np1.shape[2]):
                            pred0 = preds_np0[0, i, j] + 1  # 只统计了中间一个概率。
                            pred1 = preds_np1[0, i, j] + 1  # 只统计了中间一个概率。
                            pred2 = preds_np2[0,i,j] + 1
                            if msk_np[i, j] == 1:
                                if idx == 1:  # 说明是第一个cycle，创建包含几个reads的列表 ，
                                    listacgt.append(dictacgt[pred0])
                                    # ist_p_cyc=outputs_np_softmax[0,:,i,j].tolist()
                                    # list_p_reads.append(list_p_cyc)
                                    # list_p_total.append(list_p_reads)
                                    # list_p_reads = []

                                    # listacgt[-1] = dictacgt[preds[i][j]]
                                elif idx == 100:  # 说明是第一个cycle，创建包含几个reads的列表 ，
                                    listacgt[listacgt_idx + len_point] = listacgt[listacgt_idx + len_point] + dictacgt[pred2]  #
                                    # list_p_total[listacgt_idx+len_point].append(outputs_np_softmax[0,:,i,j].tolist())
                                    listacgt_idx = listacgt_idx + 1
                                else:  # 第二个cycle以后依次写入：
                                    listacgt[listacgt_idx + len_point] = listacgt[listacgt_idx + len_point] + dictacgt[pred1] #
                                    # list_p_total[listacgt_idx+len_point].append(outputs_np_softmax[0,:,i,j].tolist())
                                    listacgt_idx = listacgt_idx + 1
                    print(idx)
                    # print(listacgt[:2])
                    # print(list_p_total[:2])
                len_point = len(listacgt)

    filename1 = 'ACGT08_3_onebyone.json'
    with open(filename1, 'w') as file_obj:
        json.dump(listacgt, file_obj)

    # filename2 = 'possibility08.json'
    # with open(filename2, 'w') as file_obj2:
    #     json.dump(list_p_total, file_obj2)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    fastQ.writeFq('fastq/fast21_1_{}.fq'.format(timestr), listacgt, 'ROO1C001')

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

