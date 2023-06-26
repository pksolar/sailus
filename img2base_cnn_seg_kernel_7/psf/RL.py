import numpy as np
import cv2

def richardson_lucy_blind_deconvolution(image, iterations=30, psf_size=15):
    # 初始化估计的PSF为均匀分布
    estimated_psf = np.ones((psf_size, psf_size)) / (psf_size * psf_size)

    # 初始估计的原始图像
    estimated_original = image.copy()

    for _ in range(iterations):
        # 用当前估计的PSF对原始图像进行卷积
        convoluted = cv2.filter2D(estimated_original, -1, estimated_psf)

        # 计算卷积结果与观察到的图像之间的比值
        ratio = image / (convoluted + 1e-10)

        # 用当前估计的PSF对比值进行卷积
        ratio_conv = cv2.filter2D(ratio, -1, estimated_psf)

        # 更新原始图像估计
        estimated_original *= ratio_conv

        # 用比值对当前估计的PSF进行卷积
        psf_conv = cv2.filter2D(estimated_psf, -1, ratio)

        # 更新PSF估计
        estimated_psf *= psf_conv
        estimated_psf /= np.sum(estimated_psf)  # 归一化

    return estimated_original, estimated_psf

# 加载图像
image = cv2.imread('a.tif', cv2.IMREAD_GRAYSCALE)
image = image.astype(np.float32)

# 估计原始图像和PSF
estimated_original, estimated_psf = richardson_lucy_blind_deconvolution(image)

# 显示结果
cv2.imwrite("psf_rl.tif",estimated_psf)
cv2.imwrite("Estimated_rl.tif",estimated_psf)
cv2.imshow('Estimated Original', estimated_original)
cv2.imshow('Estimated PSF', estimated_psf)
cv2.waitKey(0)
cv2.destroyAllWindows()
