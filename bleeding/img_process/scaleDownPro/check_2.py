import numpy as np

def create_gaussian_kernel(size, sigma):
    kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-(size-1)/2)**2 + (y-(size-1)/2)**2)/(2*sigma**2)), (size, size))
    kernel /= np.sum(kernel)
    return kernel

# 创建参数为0.5的5x5高斯核
size = 3
sigma = 0.5
gaussian_kernel = create_gaussian_kernel(size, sigma)

print(gaussian_kernel)
