import numpy as np
import glob
import os
"""
取old mask和new mask，将找到，old为1，new为-1时，不仅生成新的为1，且还要在cycle间将label改成old的label的值。

"""

old_mask = np.load(r"E:\data\resize_test\08_resize_ori\res_deep_intent\Lane01\deepLearnData\R001C001_mask.npy")
new_mask = np.load(r"E:\data\autoupdate\msk\08hu_R001C001_msk.npy")
# old_mask = np.load(r"E:\data\resize_test\08_resize_ori\res_deep_intent\Lane01\deepLearnData\R001C001_mask.npy")
# new_mask = np.load(r"E:\code\python_PK\channelAttention-2-finetune\fastq\08h_R001C001_99.029_20230607-171956\Lane01\deepLearnDataUpdate\R001C001_mask.npy")

bingji_mask_bool = np.logical_or(old_mask==1,new_mask==1)
bingji_mask = np.where(bingji_mask_bool,1,new_mask)

# os.makedirs(rf"E:\code\python_PK\channelAttention-2-finetune\fastq\bingjilabel\Lane01\deepLearnDataUpdate",exist_ok=True)

# t1 = bingji_mask - old_mask
# t2 = bingji_mask - new_mask
# a = np.count_nonzero(t1)
# b = np.count_nonzero(t2)
# print("ddd")

indices = np.where(np.logical_and(old_mask == 1, new_mask == -1))
indices_array = np.array(indices)

print("ddd")
#通过位置信息

old_labels = glob.glob(r"E:\data\resize_test\08_resize_ori\res_deep_intent\Lane01\deepLearnData\*\label\R001C001_label.npy")
new_labels = glob.glob(r"E:\code\python_PK\img2base_cnn_seg_featureMapNoSupervised\fastq\08h_R001C001_98.404_0611152213\Lane01\deepLearnData_nobing\*\label\R001C001_label.npy")
for oldp,newp in zip(old_labels,new_labels):
    cycname = oldp.split("\\")[-3]
    old_label = np.load(oldp)
    new_label = np.load(newp)
    new_label[indices_array[0, :], indices_array[1, :]] = old_label[indices_array[0, :], indices_array[1, :]]
    os.makedirs(rf"E:\code\python_PK\img2base_cnn_seg_featureMapNoSupervised\fastq\08h_R001C001_98.404_0611152213\Lane01\bingji\{cycname}\label\\",exist_ok=True)
    np.save(rf"E:\code\python_PK\img2base_cnn_seg_featureMapNoSupervised\fastq\08h_R001C001_98.404_0611152213\Lane01\bingji\{cycname}\label\R001C001_label.npy",new_label)
    print(cycname)

np.save(r"E:\code\python_PK\img2base_cnn_seg_featureMapNoSupervised\fastq\08h_R001C001_98.404_0611152213\Lane01\bingji\R001C001_mask.npy",bingji_mask)

