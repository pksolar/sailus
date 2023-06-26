import numpy as np
import cv2
import glob
import os
#.39   56
paths_x = glob.glob(r"\\192.168.7.117\e\NGS\OutFile\202303081523_Pro017_A_SE100_5pM_originbuffer_flow.1.9.39.oldcleave_ecoli_SRM22302200035__\Image\Lane01\*\R001C100_A.tif")
save_dir = "xuan1.9.39"
i  = 0
for path in paths_x:
    name = path.split("\\")[-2]
    aA = cv2.imread(path, 0)

    aC = cv2.imread(path.replace("_A.", "_C."), 0)

    aG = cv2.imread(path.replace("_A.", "_G."), 0)

    aT = cv2.imread(path.replace("_A.", "_T."), 0)

    if not os.path.exists("{}/Lane01/{}".format(save_dir, name)):
        os.makedirs("{}/Lane01/{}".format(save_dir, name))
    cv2.imwrite("{}/Lane01/{}/R001C003_A.tif".format(save_dir, name), aA)
    cv2.imwrite("{}/Lane01/{}/R001C003_C.tif".format(save_dir, name), aC)
    cv2.imwrite("{}/Lane01/{}/R001C003_G.tif".format(save_dir, name), aG)
    cv2.imwrite("{}/Lane01/{}/R001C003_T.tif".format(save_dir, name), aT)
    i += 1
    print(i)