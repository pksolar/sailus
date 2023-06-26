import cv2
import numpy as np
import glob
import os

img_temp = np.zeros([2160,512],np.uint8)
img_temp1 = img_temp.copy()
img = cv2.imread(r"E:\code\python_PK\pythonProject\img_roi\s0\Lane01\\R001C001_A.tif",0)
img_temp1 = img[:,0:512].copy()
img[:,0:512] = 1
img[:,3*512:4*512] = img_temp1
cv2.imshow("pic",img)
cv2.waitKey(0)

