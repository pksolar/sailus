import glob

import cv2
import numpy as np
import os


#
# kernel_5_zero =  np.array([[-0.1, -0.3, -0.5, -0.3, -0.1],
#                       [-0.3,  0,     0,   0,  -0.3 ],
#                       [-0.5,  0,  4.75,  0,  -0.5],
#                       [-0.3,  0,   0,     0,  -0.3],
#                       [-0.1, -0.3, -0.5, -0.3, -0.1]])

kernel_5_2 =  np.array([[-0.1, -0.2, -0.3, -0.2, -0.1],
                      [-0.1,  -0.3,     -0.5,   -0.3,  -0.1 ],
                      [-0.3,  -0.5,  5.75,  -0.5,  -0.3],
                      [-0.1, -0.3,   -0.5,     -0.3,  -0.1],
                      [-0.1, -0.2, -0.3, -0.2, -0.1]])

paths_A = glob.glob(r"E:\data\resize_test\08_resize_ori\Lane01\*\R001C001_A.tif")
import threading

def max_minus(img_path,path_save,str_):
    gray = cv2.imread(img_path, 0)
    window_size = 3
    # 对图像进行遍历和处理
    height, width = gray.shape

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            window = gray[i - 1:i + 2, j - 1:j + 2]  # 取3x3窗口
            center = window[1, 1]  # 中心像素点的值
            max_val = np.max(window)  # 窗口内最亮值
            if center == max_val:
                continue  # 不改变中心像素点的亮度
            else:
                delta = 0.1 * max_val  # 计算亮度变化值
                gray[i, j] -= delta  # 修改中心像素点的亮度
    print(os.path.join(path_save, imgnameA.replace("_A", str_)))
    cv2.imwrite(os.path.join(path_save, imgnameA.replace("_A", str_)), gray)


def convolve(img_path, kernel,path_save,str_):
    img = cv2.imread(img_path, 0)
    dst = cv2.filter2D(img, -1, kernel).astype(float)+5
    dst = np.minimum(img,dst)
    dst = cv2.GaussianBlur(dst, (3, 3), 0.9).clip(0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(path_save, imgnameA.replace("_A", str_)), dst)

def convolve(img_path, kernel, path_save, str_):
    img = cv2.imread(img_path, 0)
    dst = cv2.filter2D(img, -1, kernel).astype(float) + 5
    dst = np.minimum(img, dst)
    dst = cv2.GaussianBlur(dst, (3, 3), 0.9).clip(0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(path_save, imgnameA.replace("_A", str_)), dst)
    #return dst

# kernel_5_2 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
def gauss_mul(img_path,  path_save, str_):
    img = cv2.imread(img_path, 0)
    dst = cv2.GaussianBlur(img, (3, 3), 0.9).clip(0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(path_save, imgnameA.replace("_A", str_)), dst)

threads = []

for pathA in paths_A:
    imgnameA = pathA.split("\\")[-1].replace(pathA.split("\\")[-1].split(".")[0], "R001C001_A")
    cycname = pathA.split("\\")[-2]
    print(cycname, " ", imgnameA)
    path_save = r"E:\data\resize_testImage\08_max_minus_gauss\Lane01\{}".format(cycname)
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    # cv2.imwrite(os.path.join(path_save, imgnameA), dstA + 2)
    # cv2.imwrite(os.path.join(path_save, imgnameA.replace("_A", "_C")), dstC + 2)
    # cv2.imwrite(os.path.join(path_save, imgnameA.replace("_A", "_G")), dstG + 2)
    # cv2.imwrite(os.path.join(path_save, imgnameA.replace("_A", "_T")), dstT + 2)

    # create threads for each channel
    threadA = threading.Thread(target=gauss_mul, args=(pathA,path_save,"_A"))
    threadC = threading.Thread(target=gauss_mul, args=(pathA.replace("R001C001_A","R001C001_C"), path_save,"_C"))
    threadG = threading.Thread(target=gauss_mul, args=(pathA.replace("R001C001_A", "R001C001_G"), path_save,"_G"))
    threadT = threading.Thread(target=gauss_mul, args=(pathA.replace("R001C001_A", "R001C001_T"), path_save,"_T"))

    # start threads
    threadA.start()
    threadC.start()
    threadG.start()
    threadT.start()

    # add threads to the list
    threads.append(threadA)
    threads.append(threadC)
    threads.append(threadG)
    threads.append(threadT)

# wait for all threads to finish
for thread in threads:
    thread.join()




