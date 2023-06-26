import numpy as np
import glob
import cv2
import os
from scipy.ndimage.filters import gaussian_filter


"""

"""
# kernel_5 =  np.array([[ 0.0187,-0.6683,  -0.5481,-0.6683,0.0187],
#                       [-0.2752,-1.0000, -1.0000,-1.0000,-0.2752],
#                       [ -0.4813, -0.5458,  3.9332,-0.5458,-0.4813],
#                       [-0.2752, -1.0000, -1.0000, -1.0000, -0.2752],
#                       [ 0.0187,-0.6683,  -0.5481,-0.6683,0.0187]])


kernel_winner = np.array([[ -0.75, -0.5, -0.75],
                     [-0.5,  5.25, -0.5],
                        [ -0.75, -0.5, -0.75]])


kernel_5_2 =  np.array([[-0.1, -0.2, -0.3, -0.2, -0.1],
                      [-0.1,  -0.3,     -0.5,   -0.3,  -0.1 ],
                      [-0.3,  -0.5,  7,  -0.5,  -0.3],
                      [-0.1, -0.3,   -0.5,     -0.3,  -0.1],
                      [-0.1, -0.2, -0.3, -0.2, -0.1]])
print(np.sum(kernel_5_2))


kernel_one = np.array([[ 0, 0, 0],
                     [0,  1, 0],
                        [ 0, 0, 0]])

paths_x = glob.glob("E:\code\python_PK\callbase\datasets\highdens_22_ori\Lane01\*\R001C001_A.tif")
save_dir = "result22_noback_mykernel_7"
for path in paths_x:
    name = path.split("\\")[-2]
    aA = cv2.imread(path,0)
    aC = cv2.imread(path.replace("_A","_C"),0)
    aG = cv2.imread(path.replace("_A","_G"),0)
    aT = cv2.imread(path.replace("_A","_T"),0)

    kernel =  cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    aA_OPEN = cv2.morphologyEx(aA,cv2.MORPH_OPEN,kernel).astype(float)
    aA = aA-aA_OPEN
    aA = cv2.GaussianBlur(aA, (3,3), 0.9)
    aA = cv2.filter2D(aA, -1, kernel_5_2).clip(0,255).astype(np.uint8)

    aC_OPEN = cv2.morphologyEx(aC, cv2.MORPH_OPEN, kernel).astype(float)
    aC = aC-aC_OPEN
    aC = cv2.GaussianBlur(aC, (3, 3), 0.9)
    aC = cv2.filter2D(aC, -1, kernel_5_2).clip(0,255).astype(np.uint8)

    aG_OPEN = cv2.morphologyEx(aG, cv2.MORPH_OPEN, kernel).astype(float)
    aG = aG - aG_OPEN
    aG = cv2.GaussianBlur(aG, (3, 3), 0.9)
    aG = cv2.filter2D(aG, -1, kernel_5_2).clip(0,255).astype(np.uint8)

    aT_OPEN = cv2.morphologyEx(aT, cv2.MORPH_OPEN, kernel).astype(float)
    aT = aT - aT_OPEN
    aT = cv2.GaussianBlur(aT, (3, 3), 0.9)
    aT = cv2.filter2D(aT, -1, kernel_5_2).clip(0,255).astype(np.uint8)
    # cv2.imshow("A",aA)
    # cv2.waitKey(0)

    # outC = cv2.filter2D(aC, -1, kernelc)
    # outG = cv2.filter2D(aG, -1, kernelg)
    # outT = cv2.filter2D(aT, -1, kernelt)




    if not os.path.exists("{}/Lane01/{}".format(save_dir,name)):
        os.makedirs("{}/Lane01/{}".format(save_dir,name))
    cv2.imwrite("{}/Lane01/{}/R001C001_A.tif".format(save_dir,name),aA)
    cv2.imwrite("{}/Lane01/{}/R001C001_C.tif".format(save_dir,name), aC)
    cv2.imwrite("{}/Lane01/{}/R001C001_G.tif".format(save_dir,name), aG)
    cv2.imwrite("{}/Lane01/{}/R001C001_T.tif".format(save_dir,name), aT)
    # cv2.imwrite("resultimg_noback/Lane01/{}/R001C001_C.tif".format(name), outC)
    # cv2.imwrite("resultimg_noback/Lane01/{}/R001C001_G.tif".format(name), outG)
    # cv2.imwrite("resultimg_noback/Lane01/{}/R001C001_T.tif".format(name), outT)


