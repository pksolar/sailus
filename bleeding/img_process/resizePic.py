import numpy as np
import cv2
import glob
import os

paths_x = glob.glob("E:\code\python_PK\callbase\datasets\dataset17\Lane01\*\R001C078_A.tif")
save_dir = "17R1C1resize1.15"
for path in paths_x:
    name = path.split("\\")[-2]
    aA = cv2.imread(path, 0)#[320:2160-320,608:4096-608]
    #imgA = np.zeros((2160, 4096), np.uint8)
    aA = cv2.resize(aA, (4710, 2484))
    aC = cv2.imread(path.replace("_A", "_C"), 0)#[320:2160-320,608:4096-608]#[80:2160-80,152:4096-152]
    aC = cv2.resize(aC, (4710, 2484))
    aG = cv2.imread(path.replace("_A", "_G"), 0)#[320:2160-320,608:4096-608]
    aG = cv2.resize(aG, (4710, 2484))
    aT = cv2.imread(path.replace("_A", "_T"), 0)#[320:2160-320,608:4096-608]
    aT = cv2.resize(aT, (4710, 2484))

    if not os.path.exists("{}/Lane01/{}".format(save_dir, name)):
        os.makedirs("{}/Lane01/{}".format(save_dir, name))
    cv2.imwrite("{}/Lane01/{}/R001C001_A.tif".format(save_dir, name), aA)
    cv2.imwrite("{}/Lane01/{}/R001C001_C.tif".format(save_dir, name), aC)
    cv2.imwrite("{}/Lane01/{}/R001C001_G.tif".format(save_dir, name), aG)
    cv2.imwrite("{}/Lane01/{}/R001C001_T.tif".format(save_dir, name), aT)
