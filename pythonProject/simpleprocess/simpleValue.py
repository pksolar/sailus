import cv2
import math
img = cv2.imread("R001C001_A.tif",0)
a,b = img.shape
value_level = [i  for i in range(0,255,10)]
print(value_level)
cv2.waitKey(0)
for i in range(a):
    for j in range(b):
            img[i,j] = math.floor(img[i,j]/5)*5+2
cv2.imshow("result",img)
cv2.imwrite("simpleValue.tif",img)
cv2.waitKey(0)
