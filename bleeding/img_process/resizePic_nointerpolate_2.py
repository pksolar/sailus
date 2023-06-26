import numpy as np
import cv2
import glob
import os
#E:\code\python_PK\callbase\datasets\highdens_22_ori\Lane01\*\R001C001_A.tif

def resize2(img):
    # 生成黑色图像
    height = img.shape[0]
    width = img.shape[1]
    black_img = np.zeros((height * 2, width * 2), np.uint8)

    # 将读取的图像的像素值填在黑色图像的偶数行和偶数列
    black_img[::2, ::2] = img
    black_img[::2, 1::2] = img
    black_img[1::2, ::2] = img

    # 将读取的图像的像素填在黑色图像的基数行和基数列
    black_img[1::2, 1::2] = img
    return  black_img

paths_x = glob.glob(r"E:\code\python_PK\callbase\datasets\highDens_08\Image\Lane01\*\R001C001_A.tif")
save_dir = "08_resize_no_interpolate_2"


#1.35 5530,2916
i= 0
height = 4320#2700
width = 8192#5120


for path in paths_x:
    name = path.split("\\")[-2]
    imgname = path.split("\\")[-1][:8]
    aA = cv2.imread(path, 0)
    aA = resize2(aA)
    aC = cv2.imread(path.replace("_A", "_C"), 0)
    aC = resize2(aC)
    aG = cv2.imread(path.replace("_A", "_G"), 0)
    aG = resize2(aG)
    aT = cv2.imread(path.replace("_A", "_T"), 0)
    aT = resize2(aT)
    print(imgname)

    if not os.path.exists("{}/Lane01/{}".format(save_dir, name)):
        os.makedirs("{}/Lane01/{}".format(save_dir, name))
    cv2.imwrite("{}/Lane01/{}/{}_A.tif".format(save_dir, name,imgname), aA)
    cv2.imwrite("{}/Lane01/{}/{}_C.tif".format(save_dir, name,imgname), aC)
    cv2.imwrite("{}/Lane01/{}/{}_G.tif".format(save_dir, name,imgname), aG)
    cv2.imwrite("{}/Lane01/{}/{}_T.tif".format(save_dir, name,imgname), aT)
    i += 1
    print(i)
