import os

import numpy as np

a = np.load(r"E:\code\python_PK\img2base_cnn_seg\debug\img_08_R001C001_Cyc048_A_img.tif.npy")
label = np.load(r"E:\code\python_PK\img2base_cnn_seg\debug\label_08_R001C001_Cyc048_A_img.tif.npy")
msk = np.load(r"E:\code\python_PK\img2base_cnn_seg\debug\msk_08_R001C001_Cyc048_A_img.tif.npy")

# b = np.load(r"E:\code\python_PK\img2base_cnn_seg\debug\label_08_R001C001_Cyc020_A_img.tif.npy")
# b_true = np.load(r"E:\data\deep\image2base\single_element\label\08_R001C001_Cyc020_label.npy")
# if b.all() == b_true.all():
#     print("yes too")
# # c = np.load(r"E:\code\python_PK\img2base_cnn_seg\debug\msk_08_R001C001_Cyc018_A_img.tif.npy")
print("hello world")