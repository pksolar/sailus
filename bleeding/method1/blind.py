import cv2
import numpy as np

# 读取卷积图像
img = cv2.imread('R001C001_A.tif',0)

# 初始化一个随机的卷积核
kernel_size = 15
kernel = np.random.randn(kernel_size, kernel_size)

# 对卷积核进行傅里叶变换
kernel_fft = np.fft.fft2(kernel)

# 对卷积图像进行傅里叶变换
img_fft = np.fft.fft2(img)

# 计算傅里叶变换后的卷积图像除以傅里叶变换后的卷积核的逆
img_fft_deconv = np.divide(img_fft, kernel_fft)

# 对结果进行反傅里叶变换
img_deconv = np.fft.ifft2(img_fft_deconv)

# 取实部并进行灰度化
img_deconv = cv2.cvtColor(np.real(img_deconv), cv2.COLOR_BGR2GRAY)

# 更新卷积核
for i in range(10):
    # 对盲去卷积后的图像进行卷积运算，得到估计出的卷积核
    img_conv = cv2.filter2D(img_deconv, -1, kernel)

    # 对估计出的卷积核进行傅里叶变换
    kernel_fft = np.fft.fft2(kernel)

    # 计算傅里叶变换后的估计卷积核除以傅里叶变换后的卷积核的逆
    kernel_fft_deconv = np.divide(kernel_fft, img_fft)

    # 对结果进行反傅里叶变换
    kernel = np.fft.ifft2(kernel_fft_deconv).real

# 显示结果
cv2.imshow('Deblurred image', img_deconv)