import cv2
import numpy as np

# 读取图像
img = cv2.imread('image.jpg')

# 转换为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 维纳滤波
ksize = 5 # 滤波器大小
sigma = 1 # 高斯核标准差
snr = 0.5 # 信噪比
h = int(sigma * snr * 255) ** 2 # 维纳滤波参数
filtered = cv2.fastNlMeansDenoising(gray, h=h, templateWindowSize=ksize, searchWindowSize=ksize)

# 显示图像
cv2.imshow('original', img)
cv2.imshow('filtered', filtered)
cv2.waitKey(0)
cv2.destroyAllWindows()
