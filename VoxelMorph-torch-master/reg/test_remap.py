import cv2
import numpy as np
img = cv2.imread(r"E:\code\python_PK\VoxelMorph-torch-master\images\fixed_512\A_0.tif",0)
width,height = img.shape
mapx = np.array([[i*2 for i in range(width)] for j in range(height)], dtype=np.float32)
mapy = np.array([[j*2 for i in range(width)] for j in range(height)], dtype=np.float32)
dst2 = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR).astype(np.uint8)  # 尺寸缩放
print(img[8,8])
print(dst2[4,4])
cv2.imshow("dst",dst2)
cv2.waitKey(0)
