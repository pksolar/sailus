import cv2
img = cv2.imread(r"E:\code\python_PK\bleeding\img_process\dst_uevne.tif",0)
cv2.imwrite("22_uneven.tif",img)