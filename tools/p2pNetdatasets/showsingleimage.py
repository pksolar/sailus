import cv2
import numpy as np
img_to_draw = cv2.imread("image/img1812.jpg")
arraypoint = np.load("point/array1812.npy")

print("hh")
for i in arraypoint:
    print("here")
    img_to_draw = cv2.circle(img_to_draw, (i[0],i[1]), 1, (0, 0, 255), -1)
cv2.imshow("draw",img_to_draw)
cv2.waitKey(0)
cv2.imwrite("draw.jpg",img_to_draw)
