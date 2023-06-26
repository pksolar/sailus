import cv2
import numpy as np

# 读入图像
gray = cv2.imread('R001C001_T.tif',0)

# 将图像转换为灰度图
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 定义窗口大小
window_size = 3

# 对图像进行遍历和处理
height, width = gray.shape
for i in range(1, height-1):
    for j in range(1, width-1):
        window = gray[i-1:i+2, j-1:j+2]  # 取3x3窗口
        center = window[1, 1]  # 中心像素点的值
        max_val = np.max(window)  # 窗口内最亮值
        if center == max_val:
            continue  # 不改变中心像素点的亮度
        else:
            delta = 0.1 * max_val  # 计算亮度变化值
            gray[i, j] -= delta  # 修改中心像素点的亮度

# 显示处理后的图像
cv2.imwrite("Tprocessed.tif",gray)
cv2.imshow('Processed Image', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()