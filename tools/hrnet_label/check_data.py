"""
check if data is right

"""
import numpy as np

a =  np.load(r"E:\code\python_PK\tools\CNNlabel_img2base\img2base\imgdata_full/Cyc022.npy")
b = np.load(r"E:\code\python_PK\tools\CNNlabel_img2base\img2base\label_full/Cyc022.npy")
c = np.load("img2base/mask/0005.npy")
t = a * c
y = b * c
print("heo")