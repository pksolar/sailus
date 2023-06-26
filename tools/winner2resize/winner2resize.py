import numpy as np
import cv2
kernel = np.array([[-0.75,-0.5,-0.75],[-0.5,5.25,-0.5],[-0.75,-0.5,-0.75]])
a = cv2.imread("ori.tif",0)
dsta = cv2.filter2D(a, -1, kernel).astype(np.uint8)
cv2.imwrite("oridst.tif",dsta)
b = cv2.imread("1.25.tif",0)
dstb = cv2.filter2D(b,-1,kernel).astype(np.uint8)
cv2.imwrite("1.25dst.tif",dstb)
