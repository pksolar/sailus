import numpy as np

# 定义矩阵的行列数
rows = 2160
cols = 4096

# 使用 meshgrid 函数生成 y 坐标矩阵和 x 坐标矩阵
y, x = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')

# 打印 y 坐标矩阵和 x 坐标矩阵
print("y 坐标矩阵：\n", y)
print("x 坐标矩阵：\n", x)