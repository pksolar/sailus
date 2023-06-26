import cv2
import matplotlib.pyplot as plt
import numpy as np
def def_equalizehist(img, L=256):

    h, w = img.shape
    # 计算图像的直方图，即存在的每个灰度值的像素点数量
    hist = cv2.calcHist([img], [0], None, [256], [0, 255])
    # 计算灰度值的像素点的概率，除以所有像素点个数，即归一化
    hist[0:255] = hist[0:255] / (h * w)
    # 设置Si
    sum_hist = np.zeros(hist.shape)
    # 开始计算Si的一部分值，注意i每增大，Si都是对前i个灰度值的分布概率进行累加
    for i in range(256):
        sum_hist[i] = sum(hist[0:i + 1])
    equal_hist = np.zeros(sum_hist.shape)
    # Si再乘上灰度级，再四舍五入
    for i in range(256):
        equal_hist[i] = int(((L - 1) - 0) * sum_hist[i] + 0.5)
    equal_img = img.copy()
    # 新图片的创建
    for i in range(h):
        for j in range(w):
            equal_img[i, j] = equal_hist[img[i, j]]

    equal_hist = cv2.calcHist([equal_img], [0], None, [256], [0, 256])
    equal_hist[0:255] = equal_hist[0:255] / (h * w)
    cv2.imshow("inverse", equal_img)
    # 显示最初的直方图
    # plt.figure("原始图像直方图")
    plt.plot(hist, color='b')
    plt.show()
    # plt.figure("直方均衡化后图像直方图")
    plt.plot(equal_hist, color='r')
    plt.show()
    cv2.waitKey()
    # return equal_hist
    return [equal_img, equal_hist]

img = cv2.imread("img1_1.tif",0)
def_equalizehist(img)
