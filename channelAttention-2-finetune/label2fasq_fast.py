import glob
import fastQ
import numpy as np

dictacgt = {1:"A",2:"C",3:"G",4:"T",5:"N"}


labels = glob.glob(r"E:\code\python_PK\channelAttention-2-finetune\fastq\bingjilabel\Lane01\deepLearnDataUpdate\*\label\*.npy")
mask = np.load(r"E:\code\python_PK\channelAttention-2-finetune\fastq\bingjilabel\Lane01\deepLearnDataUpdate\R001C001_mask.npy")
mask_faltten = mask.flatten()
indice = np.where(mask_faltten !=0)


listacgt  =[]
for labelp in labels:
    cycname = labelp.split("\\")[-3]
    print(cycname)
    label = np.load(labelp)
    label_flatten = label.flatten()
    label_pick = label_flatten[label_flatten!=0]
    rows = label_pick.shape
    for i,ele in enumerate(label_pick):
                if cycname == "Cyc001":  # 说明是第一个cycle，创建包含几个reads的列表 ，
                    listacgt.append(dictacgt[ele])
                else:  # 第二个cycle以后依次写入：3
                    listacgt[i] = listacgt[i] + dictacgt[ele]
    # if cycname == "Cyc010":
    #     break
fastQ.writeFq("fastq/labelfast_fast/"+'Lane01_fastq_notbing.fq', listacgt, 'R001C001')

