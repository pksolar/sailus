import numpy as np
import cv2
import glob
import os
#E:\code\python_PK\callbase\datasets\highdens_22_ori\Lane01\*\R001C001_A.tif

paths_x = glob.glob(r"E:\code\python_PK\bleeding\img_process\scaleDownPro\yichuan_dange_08_89.86_100\image\Lane01\*\R001C001_A.tif")
save_dir = "08_Juan_scup_1.25_08_89.86_100"


#1.35 5530,2916
i= 0
height = 2700#2700
width = 5120#5120


for path in paths_x:
    name = path.split("\\")[-2]
    imgname = path.split("\\")[-1][:8]
    aA = cv2.imread(path, 0)
    aA = cv2.resize(aA, (width, height))
    aC = cv2.imread(path.replace("_A", "_C"), 0)
    aC = cv2.resize(aC, (width, height))
    aG = cv2.imread(path.replace("_A", "_G"), 0)
    aG = cv2.resize(aG, (width, height))
    aT = cv2.imread(path.replace("_A", "_T"), 0)
    aT = cv2.resize(aT,(width, height))
    print(imgname)

    if not os.path.exists("{}/Lane01/{}".format(save_dir, name)):
        os.makedirs("{}/Lane01/{}".format(save_dir, name))
    cv2.imwrite("{}/Lane01/{}/{}_A.tif".format(save_dir, name,imgname), aA)
    cv2.imwrite("{}/Lane01/{}/{}_C.tif".format(save_dir, name,imgname), aC)
    cv2.imwrite("{}/Lane01/{}/{}_G.tif".format(save_dir, name,imgname), aG)
    cv2.imwrite("{}/Lane01/{}/{}_T.tif".format(save_dir, name,imgname), aT)
    i += 1
    print(i)
