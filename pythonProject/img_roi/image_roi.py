import cv2
import numpy as np
img_msk = np.zeros([2160,4096],np.uint8)
a,b = img_msk.shape
img_ori = cv2.imread("R001C001_A.tif",0)
center_x = int(a / 2)
center_y = int(b / 2)
print(a,b)
print(center_x,center_y)
win_w = int(1028)
win_h = int(1028)
print(center_y-win_h/2,center_y+win_h/2,center_x-win_w/2,center_x+win_w/2)

img_msk[int(center_x-win_w/2):int(center_x+win_w/2),int(center_y-win_h/2):int(center_y+win_h/2)] = 255

# _,thresh = cv2.threshold(img_msk,100,255,cv2.THRESH_BINARY)
img_and = cv2.bitwise_and(img_ori,img_msk)
img_msk[img_msk==0] = 1
img_msk[img_msk==255] = 0
img_and  = img_and + img_msk
# print(a,b)
cv2.namedWindow("msk",0)
cv2.imshow("msk",img_and)
cv2.moveWindow("msk",100,100)
cv2.imwrite("msk.tif",img_and)
cv2.waitKey(0)
