import numpy as np
from scipy.ndimage import gaussian_filter

sigma = 1.2
size = 5
center = size // 2
x, y = np.meshgrid(np.arange(size) - center, np.arange(size) - center)
kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
kernel /= kernel.sum()
rate = 1/kernel[2,2]
kernel = rate*kernel
# small_matrix = np.array([[0.06217652, 0.17620431, 0.2493522,1 0.17620431, 0.06217652]
#  [0.17620431 0.49935179 0.70664828 0.49935179 0.17620431]
#  [0.24935221 0.70664828 1.         0.70664828 0.24935221]
#  [0.17620431 0.49935179 0.70664828 0.49935179 0.17620431]
#  [0.06217652 0.17620431 0.24935221 0.17620431 0.06217652]])
# 生成随机的2160x4096矩阵
matrix = np.load("label.npy")
# 将数字为1的位置标记为1，其余位置标记为0
ones_matrix = np.where(matrix == 1, 1, 0).astype(np.float64)

# 将数字为2的位置标记为1，其余位置标记为0
twos_matrix = np.where(matrix == 2, 1, 0)

# 将数字为3的位置标记为1，其余位置标记为0
threes_matrix = np.where(matrix == 3, 1, 0)

# 将数字为4的位置标记为1，其余位置标记为0
fours_matrix = np.where(matrix == 4, 1, 0)


indices = np.argwhere(ones_matrix == 1)

# 将小矩阵中的值与大矩阵对应位置的值进行比较，并更新大矩阵的值
for index in indices:
    i, j = index
    start_i, start_j = max(i-2, 0), max(j-2, 0)
    end_i, end_j = min(i+3, ones_matrix.shape[0]), min(j+3, ones_matrix.shape[1])
    patch = kernel[start_i-i:end_i-i, start_j-j:end_j-j]
    big_matrix_patch = ones_matrix[start_i:end_i, start_j:end_j]
    updated_values = np.where(patch > big_matrix_patch, patch, big_matrix_patch)
    ones_matrix[start_i:end_i, start_j:end_j] = updated_values
    print(updated_values)
    print(ones_matrix[start_i:end_i, start_j:end_j])


# 输出结果
print("Original matrix:\n", matrix)
# print("\nOnes matrix:\n", ones_matrix_smooth)
# print("\nTwos matrix:\n", twos_matrix_smooth)
# print("\nThrees matrix:\n", threes_matrix_smooth)
# print("\nFours matrix:\n", fours_matrix_smooth)