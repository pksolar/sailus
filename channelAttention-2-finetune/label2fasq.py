import glob
import fastQ
import numpy as np

dictacgt = {1:"A",2:"C",3:"G",4:"T",5:"N"}


labels = glob.glob(r"E:\code\python_PK\channelAttention-2-finetune\fastq\bingjilabel\Lane01\deepLearnDataUpdate\*\label\*.npy")
mask = np.load(r"E:\code\python_PK\channelAttention-2-finetune\fastq\bingjilabel\Lane01\deepLearnDataUpdate\R001C001_mask.npy")
listacgt  =[]
for labelp in labels:
    cycname = labelp.split("\\")[-3]
    print(cycname)
    label = np.load(labelp)
    rows,cols  = label.shape
    for i in range(rows):
        for j in range(cols):
            if label[i,j] !=0:
                if cycname == "Cyc001":  # 说明是第一个cycle，创建包含几个reads的列表 ，
                    listacgt.append(dictacgt[label[i,j]])
                else:  # 第二个cycle以后依次写入：
                    listacgt[reads_idx] = listacgt[reads_idx] + dictacgt[label[i,j]]
    if cycname == "Cyc010":
        break
fastQ.writeFq("fastq/labelfastq/"+'Lane01_fastq.fq', listacgt, 'R001C001')

