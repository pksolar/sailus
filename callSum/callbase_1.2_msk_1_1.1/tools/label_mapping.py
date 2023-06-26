import os
import glob
import numpy as np
import fastQ
# label 里是0 1 2 3 4 5 ，0为背景，5为N
paths = glob.glob(r"E:\code\python_PK\callbase\datasets\21\Res\Lane01\deepLearnData\*\label\R001C001_label.npy")
dict_ = {1:'A',2:"C",3:"G",4:"T",5:"N"}
listacgt = []
idx = 0

for path in paths:
    idx2 = 0
    label = np.load(path).astype(int)
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            if label[i][j] != 0:
                if idx == 0: #第一个cycle创建一下。
                    listacgt.append(dict_[label[i][j]])
                else:
                    listacgt[idx2] =  listacgt[idx2] + dict_[label[i][j]]
                    idx2 = idx2 + 1
    print(idx)
    idx = idx + 1
fastQ.writeFq('../fastq/fastq_label3.fq',listacgt,'ROO1C001')

