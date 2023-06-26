import cv2
import numpy as np
def apply_mapping(img, map_x, map_y):
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
def create_custom_mapping(shape):
    map_x = np.zeros(shape, dtype=np.float32)
    map_y = np.zeros(shape, dtype=np.float32)

    for y in range(shape[0]):
        for x in range(shape[1]):
            if y % 2 == 0:
                # 偶数行
                map_x[y, x] = np.clip(x + 0.8, 0, shape[1] - 1)
                map_y[y, x] = np.clip(y + 0.8, 0, shape[0] - 1)
            else:
                # 奇数行
                map_x[y, x] = np.clip(x + 0.5, 0, shape[1] - 1)
                map_y[y, x] = np.clip(y + 0.5, 0, shape[0] - 1)

    return map_x, map_y
# 读入变形后的图像
img = cv2.imread('image.tif',0)

# 创建网格
rows, cols = img.shape[:2]
x, y = np.meshgrid(np.arange(cols), np.arange(rows))



# 创建映射
map_x, map_y = create_custom_mapping(img.shape)


# 应用映射变形图像
warped_img = apply_mapping(img, map_x, map_y)
# 计算逆映射的坐标映射矩阵
map_map_x = apply_mapping(map_x,map_x,map_y)
map_map_y = apply_mapping(map_y,map_x,map_y)

# 恢复原始图像
img_inv = cv2.remap(img, map_map_x, map_map_y, cv2.INTER_LINEAR)

# 显示结果
cv2.imshow('deformed_image', img)
cv2.imshow('recovered_image', img_inv)
cv2.imwrite('recovered_image.tif', img_inv)
cv2.waitKey(0)
cv2.destroyAllWindows