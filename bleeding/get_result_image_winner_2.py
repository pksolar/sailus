import glob

import cv2
import numpy as np
import os

kernelA = np.array([[ -0.75, -0.5, -0.75],
          [-0.5,  5.25, -0.5],
          [ -0.75, -0.5, -0.75]])
kernelC = np.array([[ -0.75, -0.5, -0.75],
          [-0.5,  5.25, -0.5],
          [ -0.75, -0.5, -0.75]])
kernelG = np.array([[ 0.4952, -0.9764,  0.5540],
          [-0.9867,  2.5496, -0.9115],
          [ 0.5849, -0.9357,  0.4773]])
kernelT = np.array([[ -0.75, -0.5, -0.75],
          [-0.5,  5.25, -0.5],
          [ -0.75, -0.5, -0.75]])

kernel_5_zero =  np.array([[-0.1, -0.3, -0.5, -0.3, -0.1],
                      [-0.3,  0,     0,   0,  -0.3 ],
                      [-0.5,  0,  4.75,  0,  -0.5],
                      [-0.3,  0,   0,     0,  -0.3],
                      [-0.1, -0.3, -0.5, -0.3, -0.1]])

kernel_5_2 =  np.array([[-0.1, -0.2, -0.3, -0.2, -0.1],
                      [-0.1,  -0.3,     -0.5,   -0.3,  -0.1 ],
                      [-0.3,  -0.5,  5.75,  -0.5,  -0.3],
                      [-0.1, -0.3,   -0.5,     -0.3,  -0.1],
                      [-0.1, -0.2, -0.3, -0.2, -0.1]])

winnerkernel = np.array([[ -0.75, -0.5, -0.75],
          [-0.5,  5.25, -0.5],
          [ -0.75, -0.5, -0.75]])

# dst5 = cv2.filter2D(input2,-1,kernel_5_zero).astype(float)
#
# dst5 = cv2.GaussianBlur(dst5,(3,3),0.9).clip(0,255).astype(np.uint8)


paths_A = glob.glob(r"E:\data\resize_test\08_resize_ori\Lane01\*\R001C001_A.tif")
for pathA in paths_A:
    imgnameA = pathA.split("\\")[-1].replace(pathA.split("\\")[-1].split(".")[0], "R001C001_A")
    imgA = cv2.imread(pathA,0)
    dstA = cv2.filter2D(imgA, -1, kernel_5_2).astype(float)
    dstA = cv2.GaussianBlur(dstA, (3, 3), 0.9).clip(0, 255).astype(np.uint8)

    imgC = cv2.imread(pathA.replace("R001C001_A","R001C001_C"),0)
    dstC = cv2.filter2D(imgC, -1, kernel_5_2).astype(float)
    dstC = cv2.GaussianBlur(dstC, (3, 3), 0.9).clip(0, 255).astype(np.uint8)

    imgG= cv2.imread(pathA.replace("R001C001_A", "R001C001_G"),0)
    dstG = cv2.filter2D(imgG, -1, kernel_5_2).astype(float)
    dstG = cv2.GaussianBlur(dstG, (3, 3), 0.9).clip(0, 255).astype(np.uint8)

    imgT = cv2.imread(pathA.replace("R001C001_A", "R001C001_T"),0)
    dstT = cv2.filter2D(imgT, -1, kernel_5_2).astype(float)
    dstT = cv2.GaussianBlur(dstT, (3, 3), 0.9).clip(0, 255).astype(np.uint8)

    cycname = pathA.split("\\")[-2]
    #imgnameA = pathA.split("\\")[-1].replace("R003C100","R001C001")
    print(cycname," ",imgnameA)
    path_save= r"E:\data\resize_testImage\08_bleed5_5.75\Lane01\{}".format(cycname)
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    cv2.imwrite(os.path.join(path_save,imgnameA),dstA+2)
    cv2.imwrite(os.path.join(path_save, imgnameA.replace("_A","_C")), dstC+2)
    cv2.imwrite(os.path.join(path_save, imgnameA.replace("_A","_G")), dstG+2)
    cv2.imwrite(os.path.join(path_save, imgnameA.replace("_A","_T")), dstT+2)


