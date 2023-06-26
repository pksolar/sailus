import numpy as np
import glob
import os
"""
取old mask和new mask，将找到，old为1，new为-1时，不仅生成新的为1，且还要在cycle间将label改成old的label的值。

"""
old_dir = r"E:\data\resize_test\08_resize_ori\res_deep_intent\Lane01\deepLearnData\\"
new_dir = r"E:\code\python_PK\img2base_cnn_seg_featureMapNoSupervised\fastq\08h_R001C001_98.404_0613121439\Lane01\deepLearnData\\"
bing_dir = new_dir.replace("deepLearnData","bingji")
old_mask = np.load(old_dir+"R001C001_mask.npy")
new_mask = np.load(new_dir+"R001C001_mask.npy")

bingji_mask_bool = np.logical_or(old_mask==1,new_mask==1)
bingji_mask = np.where(bingji_mask_bool,1,new_mask)

indices = np.where(np.logical_and(old_mask == 1, new_mask == -1))
indices_array = np.array(indices)

old_labels = glob.glob(old_dir+"\*\label\R001C001_label.npy")
new_labels = glob.glob(new_dir+"\*\label\R001C001_label.npy")
for oldp,newp in zip(old_labels,new_labels):
    cycname = oldp.split("\\")[-3]
    old_label = np.load(oldp)
    new_label = np.load(newp)
    new_label[indices_array[0, :], indices_array[1, :]] = old_label[indices_array[0, :], indices_array[1, :]]
    os.makedirs(rf"E:\code\python_PK\img2base_cnn_seg_featureMapNoSupervised\fastq\08h_R001C001_98.404_0613121439\Lane01\bingji\{cycname}\label\\",exist_ok=True)
    np.save(rf"E:\code\python_PK\img2base_cnn_seg_featureMapNoSupervised\fastq\08h_R001C001_98.404_0613121439\Lane01\bingji\{cycname}\label\R001C001_label.npy",new_label)
    print(cycname)

np.save(r"E:\code\python_PK\img2base_cnn_seg_featureMapNoSupervised\fastq\08h_R001C001_98.404_0613121439\Lane01\bingji\R001C001_mask.npy",bingji_mask)

