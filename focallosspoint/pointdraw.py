import cv2
import numpy as np

label = np.load(r"E:\code\python_PK\callbase\datasets\30\Res\Lane01\deepLearnData1\Cyc001\label\R001C001_label.npy").astype(int)
img = cv2.imread("3channel.jpg")
for i in range(label.shape[0]):
    for j in range(label.shape[1]):
        if label[i,j] == 1: #bgr
            img_to_draw = cv2.circle(img, (j,i), 1, (0, 255, 0), -1) #绿色  A
        elif  label[i,j] == 2:
            img_to_draw = cv2.circle(img, (j, i), 1, (255, 0, 0), -1)#蓝色 C
        elif label[i, j] == 3:
            img_to_draw = cv2.circle(img, (j, i), 1, (0, 0, 255), -1) #红色 G
        elif label[i, j] == 4:
            img_to_draw = cv2.circle(img, (j, i), 1, (0, 255, 255), -1)# 黄色 T
        elif label[i, j] == 5:
            img_to_draw = cv2.circle(img, (j, i), 1, (0, 0, 0), -1)


cv2.imshow("here",img_to_draw)
cv2.imwrite("pointdraw.jpg",img_to_draw)
cv2.waitKey(0)