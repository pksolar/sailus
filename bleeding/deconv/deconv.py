import numpy as np
import cv2

def blind_deconvolution(image, psf, max_iter):
    # 初始化估计的图像和模糊核
    estimate = image
    kernel = psf

    for i in range(max_iter):
        # 估计模糊核
        error = cv2.filter2D(estimate, -1, kernel) - image
        error_fft = np.fft.fft2(error)
        estimate_fft = np.fft.fft2(estimate)
        kernel_fft = error_fft / (estimate_fft + 1e-6)
        kernel = np.abs(np.fft.ifft2(kernel_fft))

        # 估计原始图像
        image_fft = np.fft.fft2(image)
        kernel_fft = np.fft.fft2(kernel, s=image.shape)
        estimate_fft = image_fft / (kernel_fft + 1e-6)
        estimate = np.abs(np.fft.ifft2(estimate_fft))

    return estimate, kernel

# 读取图像
image = cv2.imread('blur.tif', 0)

# 创建一个初始的模糊核
psf = np.ones((5, 5)) / 25

# 执行盲去卷积
estimate, kernel = blind_deconvolution(image, psf, max_iter=20)

# 显示去卷积后的图像
cv2.imshow('Deblurred', estimate)
cv2.waitKey(0)
cv2.destroyAllWindows()
