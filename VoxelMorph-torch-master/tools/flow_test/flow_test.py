import numpy as np
import cv2

img = np.zeros((10,10,3), np.uint8)

for i in range(10):
    for j in range(10):
        img[i,j] = [i*10,j*10,(i+j)*10]

cv2.imshow('image',img)
cv2.imwrite("image.tif",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
