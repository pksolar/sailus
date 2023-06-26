import cv2
import numpy as np

# 读取图像
image = cv2.imread('1.tif', cv2.IMREAD_GRAYSCALE)

# 添加padding
padded_image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)

# 获取图像尺寸
height, width = image.shape

# 创建输出图像
output = np.zeros((height, width), dtype=np.uint8)

# 进行池化操作
for i in range(height):
    for j in range(width):
        # 提取3x3区域
        region = padded_image[i:i+3, j:j+3]
        # 计算区域内的最大值
        max_value = np.max(region)
        # 将最大值赋给输出图像对应位置
        output[i, j] = max_value

# 显示输出图像
cv2.imwrite("1_max.tif",output)
cv2.imshow('Output Image', output)
cv2.waitKey(0)
cv2.destroyAllWindows()
