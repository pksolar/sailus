import cv2
img = cv2.imread(r"E:\code\python_PK\test2.tif",0)
img2 = cv2.resize(img,[1024,1024])
cv2.imwrite("imgbig.tif",img2)