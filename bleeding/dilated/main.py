import numpy as np
import glob
import cv2
import os

paths_x = glob.glob("E:\code\python_PK\callbase\datasets\highDens_08\Image\Lane01\*\R001C001_A.tif")
save_dir = "dilated_cross_img"
kernel =  cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
# kernel_d =  cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
kernel_d =  cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
for path in paths_x:
    name = path.split("\\")[-2]
    aA = cv2.imread(path,0)
    aC = cv2.imread(path.replace("_A","_C"),0)
    aG = cv2.imread(path.replace("_A","_G"),0)
    aT = cv2.imread(path.replace("_A","_T"),0)


    aA_OPEN = cv2.morphologyEx(aA,cv2.MORPH_OPEN,kernel).astype(float)
    aA = aA-aA_OPEN
    aA = cv2.dilate(aA,kernel_d)
    aA = aA + aA_OPEN


    aC_OPEN = cv2.morphologyEx(aC, cv2.MORPH_OPEN, kernel).astype(float)
    aC = aC-aC_OPEN
    aC = cv2.dilate(aC, kernel_d)
    aC = aC + aC_OPEN


    aG_OPEN = cv2.morphologyEx(aG, cv2.MORPH_OPEN, kernel).astype(float)
    aG = aG - aG_OPEN
    aG = cv2.dilate(aG, kernel_d)
    aG = aG + aG_OPEN


    aT_OPEN = cv2.morphologyEx(aT, cv2.MORPH_OPEN, kernel).astype(float)
    aT = aT - aT_OPEN
    aT = cv2.dilate(aG, kernel_d)
    aT = aT + aT_OPEN





    if not os.path.exists("{}/Lane01/{}".format(save_dir,name)):
        os.makedirs("{}/Lane01/{}".format(save_dir,name))
    cv2.imwrite("{}/Lane01/{}/R001C001_A.tif".format(save_dir,name),aA.clip(0,255).astype(np.uint8))
    cv2.imwrite("{}/Lane01/{}/R001C001_C.tif".format(save_dir,name), aC.clip(0,255).astype(np.uint8))
    cv2.imwrite("{}/Lane01/{}/R001C001_G.tif".format(save_dir,name), aG.clip(0,255).astype(np.uint8))
    cv2.imwrite("{}/Lane01/{}/R001C001_T.tif".format(save_dir,name), aT.clip(0,255).astype(np.uint8))
    # cv2.imwrite("resultimg_noback/Lane01/{}/R001C001_C.tif".format(name), outC)
    # cv2.imwrite("resultimg_noback/Lane01/{}/R001C001_G.tif".format(name), outG)
    # cv2.imwrite("resultimg_noback/Lane01/{}/R001C001_T.tif".format(name), outT)


