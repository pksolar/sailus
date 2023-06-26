import cv2
import numpy as np
import time

def Sharpen(path):
    time_start = time.time()
    '''读取图像'''
    image = cv2.imread(path, 1)
    '''resize'''
    height = 400
    width = int(400 * image.shape[1] / image.shape[0])
    image = cv2.resize(image, (width, height))
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    ksize = 3
    sharpen_ing = np.copy(image)  # 创建输出图像
    H, W, C = sharpen_ing.shape
    # 没有计算边缘像素点数小于卷积核大小的部分
    # 所占部分很小所以可以忽略
    for h in range(0, H - ksize + 1):
        for w in range(0, W - ksize + 1):
            for c in range(0, 3):
                sharpen_ing[h, w, c] = np.sum(kernel * sharpen_ing[h:h + ksize, w:w + ksize, c])
    time_end = time.time()
    print(time_end - time_start, 's')  # 计时器
    '''show'''
    cv2.imshow('image', np.hstack((image, sharpen_ing)))
    cv2.waitKey(0)
def sobel(path):
    img = cv2.imread(path, 1)

    SobelX = cv2.Sobel(img, cv2.CV_16S, 1, 0)  # 计算 x 轴方向
    SobelY = cv2.Sobel(img, cv2.CV_16S, 0, 1)  # 计算 y 轴方向
    absX = cv2.convertScaleAbs(SobelX)  # 转回 uint8
    absY = cv2.convertScaleAbs(SobelY)  # 转回 uint8
    SobelXY = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    cv2.imshow('image',SobelXY)
    cv2.waitKey(0)


sobel("E:\code\python_PK\pythonProject\image_ori\Lane01\Cyc001\R001C001_A.tif")