import numpy as np
import cv2
import glob
import os

paths_x = glob.glob(r"F:\08machineDeepLearn_z\*\R001C001_A.tif")
save_dir = "08_deepzhou"
i  = 0
for path in paths_x:
    name = path.split("\\")[-2]
    aA = cv2.imread(path, 0)

    aC = cv2.imread(path.replace("_A.", "_C."), 0)

    aG = cv2.imread(path.replace("_A.", "_G."), 0)

    aT = cv2.imread(path.replace("_A.", "_T."), 0)

    if not os.path.exists("{}/Lane01/{}".format(save_dir, name)):
        os.makedirs("{}/Lane01/{}".format(save_dir, name))
    cv2.imwrite("{}/Lane01/{}/R001C001_A.tif".format(save_dir, name), aA)
    cv2.imwrite("{}/Lane01/{}/R001C001_C.tif".format(save_dir, name), aC)
    cv2.imwrite("{}/Lane01/{}/R001C001_G.tif".format(save_dir, name), aG)
    cv2.imwrite("{}/Lane01/{}/R001C001_T.tif".format(save_dir, name), aT)
    i += 1
    print(i)