import math
import numpy as np
import cv2



def get_motion_dsf(image_size, motion_angle, motion_dis):
    PSF = np.zeros(image_size)  # 点扩散函数
    x_center = (image_size[0] - 1) / 2
    y_center = (image_size[1] - 1) / 2

    sin_val = math.sin(motion_angle * math.pi / 180)
    cos_val = math.cos(motion_angle * math.pi / 180)

    # 将对应角度上motion_dis个点置成1
    for i in range(motion_dis):
        x_offset = round(sin_val * i)
        y_offset = round(cos_val * i)
        PSF[int(x_center - x_offset), int(y_center + y_offset)] = 1

    return PSF / PSF.sum()  # 归一化

image = cv2.imread("a.tif",0)
PSF = get_motion_dsf(image.shape, 135, 150)
dst = np.zeros(PSF.shape)
norm_psf = cv2.normalize(PSF, dst, 1.0, 0.0, cv2.NORM_MINMAX)
cv2.imwrite("psf.tif",(norm_psf*255).astype(np.uint8))
cv2.imshow('psf', (norm_psf*255).astype(np.uint8))

# def wiener(input,PSF,eps,SNR=0.001):        #维纳滤波，SNR=0.01
#     input_fft=fft.fft2(input)
#     PSF_fft=fft.fft2(PSF) +eps
#     PSF_fft_1=np.conj(PSF_fft) /(np.abs(PSF_fft)**2 + SNR)
#     result=fft.ifft2(input_fft * PSF_fft_1)
#     result=np.abs(fft.fftshift(result))
#     return result
#
# def inverse(input, PSF, eps):
#     input_fft = fft.fft2(input)
#     PSF_fft = fft.fft2(PSF) + eps #噪声功率
#     result = fft.ifft2(input_fft / PSF_fft) #计算F(u,v)的傅里叶反变换
#     result = np.abs(fft.fftshift(result))
#     return result
#
# result=wiener(blurred,PSF,1e-2)
# result = inverse(blurred, PSF,1e-3)   #逆滤波