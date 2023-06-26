import numpy as np
from scipy.ndimage import grey_dilation, generate_binary_structure

# 生成一个5x5的二维结构元素
struct = generate_binary_structure(2, 2)



# 创建一个3x3的二维数组，其中心位置为1，其他位置为0
arr = np.zeros((20, 20), dtype=np.int32)
arr[5, 5] = 1
arr[2, 2] = 1
arr[7, 7] = 1

# 对数组进行膨胀操作
dilated_arr = grey_dilation(arr, footprint=struct)
dilated_arr2 = grey_dilation(dilated_arr,footprint=struct)
# 输出膨胀前后的数组形状
print("原始数组形状：", arr.shape)
print("膨胀后数组形状：", dilated_arr.shape)