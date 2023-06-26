import glob

import cv2
import numpy as np
import os

kernelA = np.array([[ 0.6215, -1.2322,  0.6924],
          [-1.2115,  3.0225, -1.1818],
          [ 0.7205, -1.1927,  0.6173]])
kernelC = np.array([[ 0.6337, -1.1617,  0.6863],
          [-1.1668,  2.9000, -1.1820],
          [ 0.6998, -1.1587,  0.6459]])
kernelG = np.array([[ 0.5647, -1.1208,  0.6364],
          [-1.1266,  2.8461, -1.0829],
          [ 0.6598, -1.0947,  0.5696]])
kernelT = np.array([[ 0.3454, -0.9192,  0.4509],
          [-0.8886,  2.7103, -0.8827],
          [ 0.4674, -0.9256,  0.3365]])

paths_A = glob.glob("E:\code\python_PK\callbase\datasets\highDens\Image\Lane01\*\R001C001_A.tif")
for pathA in paths_A:

    imgA = cv2.imread(pathA,0)
    dstA = cv2.filter2D(imgA, -1, kernelA)

    imgC = cv2.imread(pathA.replace("_A","_C"),0)
    dstC = cv2.filter2D(imgC, -1, kernelC)

    imgG= cv2.imread(pathA.replace("_A", "_G"),0)
    dstG = cv2.filter2D(imgG, -1, kernelG)

    imgT = cv2.imread(pathA.replace("_A", "_T"),0)
    dstT = cv2.filter2D(imgT, -1, kernelT)

    cycname = pathA.split("\\")[-2]
    imgnameA = pathA.split("\\")[-1]
    print(cycname," ",imgnameA)
    path_save= "E:\code\python_PK\callbase\datasets\highDense_output\Image\Lane01\{}".format(cycname)
    if not os.path.exists(path_save):
        os.mkdir(path_save)
    cv2.imwrite(os.path.join(path_save,imgnameA),dstA)
    cv2.imwrite(os.path.join(path_save, imgnameA.replace("_A","_C")), dstC)
    cv2.imwrite(os.path.join(path_save, imgnameA.replace("_A","_G")), dstG)
    cv2.imwrite(os.path.join(path_save, imgnameA.replace("_A","_T")), dstT)


