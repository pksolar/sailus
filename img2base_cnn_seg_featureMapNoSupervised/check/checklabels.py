import  numpy as np
import matplotlib.pyplot as plt
import cv2
a = np.load(r"E:\code\python_PK\img2base_cnn_seg_featureMapNoSupervised\fastq\08h_R001C001_98.404_0613121439\Lane01\bingji\Cyc006\label\R001C001_label.npy")
b = np.load(r"E:\data\resize_test\08_resize_ori\res_deep_intent\Lane01\deepLearnData\Cyc006\label\R001C001_label.npy")
g = np.where(a!=b)
pos_array = np.array(g)
fig, ax = plt.subplots()

# 设置黑色背景
ax.set_facecolor('black')

# 绘制红色点
ax.scatter(pos_array[0, :], pos_array[1, :], c='red')

# 显示图形
plt.show()