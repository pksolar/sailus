import cv2
import numpy as np
a =  cv2.imread(r"E:\code\python_PK\callbase\datasets\highDens\Result_deepLearn\Image\Lane01\Cyc001\R001C001_A.jpg",0)
c =  cv2.imread("E:\code\python_PK\callbase\datasets\highDens\Result_deepLearn\Image\Lane01\Cyc001\R001C001_C.jpg",0)
g =  cv2.imread("E:\code\python_PK\callbase\datasets\highDens\Result_deepLearn\Image\Lane01\Cyc001\R001C001_G.jpg",0)

total = np.zeros((2160,4096,3))
total[:,:,0] = a
total[:,:,1] = c
total[:,:,2] = g
cv2.imwrite("total.jpg",total)

cv2.imshow("total",total)

cv2.waitKey(0)
