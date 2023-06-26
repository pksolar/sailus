import numpy as np
import glob
from scipy.signal import convolve2d
import time
import fastQ
"""
tensor([[[[ 0.2635,  0.1821,  0.1736],
          [ 0.0642,  0.0933,  0.3328],
          [-0.2122, -0.1043,  0.2627]]]], requires_grad=True)
loss: 0.006274381186813116 i:  1
Parameter containing:
tensor([[[[ 0.2055,  0.0616, -0.1687],
          [ 0.0633, -0.1452,  0.1974],
          [ 0.0338, -0.1171, -0.0122]]]], requires_grad=True)
loss2: 0.0029406677931547165 i:  1
Parameter containing:
tensor([[[[ 0.1486,  0.1240,  0.0913],
          [ 0.1643,  0.0587, -0.1207],
          [-0.0157, -0.2190,  0.1747]]]], requires_grad=True)
loss3: 0.006978650111705065 i:  1
Parameter containing:
tensor([[[[ 0.1525,  0.0896,  0.0437],
          [ 0.1082,  0.0134,  0.1404],
          [ 0.0591, -0.1424,  0.2938]]]], requires_grad=True)
loss4: 0.006105361972004175 i:  1

"""
kernela = np.array([[ 0.0712,  0.0748,  0.1005],
          [ 0.0667,  0.1419,  0.0696],
          [-0.0117,  0.0269,  0.0884]])
kernelc= np.array([[ 0.0344,  0.0269,  0.0240],
          [ 0.0296, -0.2476,  0.0342],
          [ 0.0286,  0.0177,  0.0183]])
kernelg= np.array([[ 0.0982,  0.0867,  0.1308],
          [ 0.0814, -0.0474,  0.0280],
          [ 0.1022, -0.0205,  0.1045]])
kernelt= np.array([[ 0.0736,  0.0608,  0.1065],
          [ 0.0794, -0.1062,  0.0839],
          [ 0.1309,  0.0309,  0.1042]])
paths_x = glob.glob("E:\code\python_PK\callbase\datasets\highDens\Result\Lane01\deepLearnData_final\*\intensity\R001C001_A.npy")
msk = np.load("E:\code\python_PK\callbase\datasets\highDens\Result\Lane01\deepLearnData_final\R001C001_mask.npy").astype(int)
totalreadsnum = sum(sum(abs(msk)))
dictacgt = {1: "A", 2: "C", 3: "G", 4: "T"}
predict_list = [""]*totalreadsnum
for path in paths_x:
    aA = np.load(path)
    aC = np.load(path.replace("_A","_C"))
    aG = np.load(path.replace("_A","_G"))
    aT = np.load(path.replace("_A","_T"))


    outA = convolve2d(aA,kernela,'same','fill')[np.newaxis,:]
    outC = convolve2d(aC,kernelc,'same','fill')[np.newaxis,:]
    outG = convolve2d(aG,kernelg,'same','fill')[np.newaxis,:]
    outT = convolve2d(aT,kernelt,'same','fill')[np.newaxis,:]

    out = np.concatenate([outA,outC,outG,outT]) #保存以后，写成fastq，去做mapping。

    pred = np.argmax(out,axis=0).astype(int)  # pred 取 0,1,2,3 #1，2,3,4


    str_acgt = ""
    k = 0
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            if msk[i,j] !=0:
                #str_acgt = str_acgt + dictacgt[pred[i,j]+1]
                predict_list[k] = predict_list[k] + dictacgt[pred[i,j]+1]
                k = k + 1
    #print(predict_list)
    print(path)

timestr = time.strftime("%Y%m%d-%H%M%S")
fastQ.writeFq(r'E:\code\python_PK\bleeding\fastq/fastq-{}.fq'.format(timestr), predict_list, 'ROO1C001')






    # if aA.all == out.all:
    #     print("same")
    # else:
    #     print("no")
    # print("wow")


