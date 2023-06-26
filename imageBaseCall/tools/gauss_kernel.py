import numpy as np

# 生成5x5的高斯核
sigma = 1.2
size = 5
center = size // 2
x, y = np.meshgrid(np.arange(size) - center, np.arange(size) - center)
kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
kernel /= kernel.sum()
rate = 1/kernel[2,2]
kernel = rate*kernel
# 输出结果
print(kernel*rate)