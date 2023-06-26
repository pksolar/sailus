import glob
import os

import fastQ
import numpy as np

dictacgt = {1:"A",2:"C",3:"G",4:"T",5:"N"}


# labels = glob.glob(r"E:\code\python_PK\img2base_cnn_seg_featureMapNoSupervised\fastq\08h_R001C001_98.404_0611152213\Lane01\bingji\*\label\*.npy")
# mask = np.load(r"E:\code\python_PK\img2base_cnn_seg_featureMapNoSupervised\fastq\08h_R001C001_98.404_0611152213\Lane01\bingji\R001C001_mask.npy")
# old_labels = glob.glob(r"E:\data\resize_test\08_resize_ori\res_deep_intent\Lane01\deepLearnData\*\label\R001C001_label.npy")
labels = glob.glob(r"E:\data\resize_test\08_resize_ori\res_deep_intent\Lane01\deepLearnData\*\label\*.npy")
mask = np.load(r"E:\data\resize_test\08_resize_ori\res_deep_intent\Lane01\deepLearnData\R001C001_mask.npy")

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
                else:  # 第二个cycle以后依次写入：
                    listacgt[i] = listacgt[i] + dictacgt[ele]
    # if cycname == "Cyc010":
    #     break
dirpath = r"E:\code\python_PK\img2base_cnn_seg_featureMapNoSupervised\fastq\08h_R001C001_98.404_0611152213\old_label\\"
os.makedirs(dirpath,exist_ok=True)
fastQ.writeFq(dirpath+'Lane01_fastq.fq', listacgt, 'R001C001')


