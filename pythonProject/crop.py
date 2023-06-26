import cv2
import numpy as np
img = cv2.imread(r"E:\code\python_PK\test.tif",0)
cv2.imwrite(r"E:\code\python_PK\test2.tif",img[:512,:512])
img = np.zeros([1024,1024],np.uint8)
cv2.imwrite("test2.tif",img)