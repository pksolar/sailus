import cv2
img = cv2.imread("../imgbig.tif",0)
img2 = cv2.resize(img,[2048,2048])
cv2.imwrite("big2048.tif",img2)