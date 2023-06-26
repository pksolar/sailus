import cv2
img = cv2.imread("R001C001_A.tif",0)
img1 = img[:,0:256]
cv2.imwrite("img1_1.tif",img1)
cv2.imwrite("img2_1.tif",img[:,3*512:int(3.5*512)])