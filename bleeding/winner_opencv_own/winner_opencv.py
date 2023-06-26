import cv2
import numpy as np

# 读取图像
img = cv2.imread('img/R001C001_A.tif',0)




# 使用高斯模糊平滑图像
blur = cv2.GaussianBlur(img, (5,5), 0)

# 使用原始图像减去模糊后的图像，得到高频部分
high_freq = cv2.subtract(img, blur)

# 将高频部分加到原始图像上
sharpened = cv2.addWeighted(img, 2, high_freq, -1, 0)

# 显示图像

cv2.imwrite("img/A_res.tif",sharpened)
# 显示图像
# cv2.imshow('original', img)
# cv2.imshow('filtered', filtered)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
