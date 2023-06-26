import numpy as np
import cv2
import glob
import os

paths_x = glob.glob("E:\code\python_PK\callbase\datasets\highDens\Image\Lane01\*\R001C001_A.tif")
save_dir = "08_resize_black_big"
for path in paths_x:
    name = path.split("\\")[-2]
    aA = cv2.imread(path, 0)[160:2160-160,304:4096-304]
    imgA = np.zeros((2160, 4096), np.uint8)
    imgA[160:2160-160,304:4096-304] = aA
    aC = cv2.imread(path.replace("_A", "_C"), 0)[160:2160-160,304:4096-304]
    imgC = np.zeros((2160, 4096), np.uint8)
    imgC[160:2160-160,304:4096-304] = aC

    aG = cv2.imread(path.replace("_A", "_G"), 0)[160:2160-160,304:4096-304]
    imgG = np.zeros((2160, 4096), np.uint8)
    imgG[160:2160-160,304:4096-304] = aG

    aT = cv2.imread(path.replace("_A", "_T"), 0)[160:2160-160,304:4096-304]
    imgT = np.zeros((2160, 4096), np.uint8)
    imgT[160:2160-160,304:4096-304] = aT


    if not os.path.exists("{}/Lane01/{}".format(save_dir, name)):
        os.makedirs("{}/Lane01/{}".format(save_dir, name))
    cv2.imwrite("{}/Lane01/{}/R001C001_A.tif".format(save_dir, name), imgA)
    cv2.imwrite("{}/Lane01/{}/R001C001_C.tif".format(save_dir, name), imgC)
    cv2.imwrite("{}/Lane01/{}/R001C001_G.tif".format(save_dir, name), imgG)
    cv2.imwrite("{}/Lane01/{}/R001C001_T.tif".format(save_dir, name), imgT)
