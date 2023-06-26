import cv2
import numpy as np


img = cv2.imread('image.tif')

rows, cols = img.shape[:2]
map_x = np.zeros((rows, cols), np.float32)
map_y = np.zeros((rows, cols), np.float32)

for i in range(rows):
    for j in range(cols):
        map_x[i, j] = j + 1  # x坐标加1
        map_y[i, j] = i + 1  # y坐标加1

dst = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)

cv2.imwrite("image_dst.tif",dst)
