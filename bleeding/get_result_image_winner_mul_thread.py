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

paths_A = glob.glob(r"E:\data\highDensity\dense0.6\R001C001\Lane01\*\R001C001_A.tif")
import threading

def convolve(img_path, kernel,path_save,str_):
    img = cv2.imread(img_path, 0)
    dst = cv2.filter2D(img, -1, kernel).astype(float)
    dst = cv2.GaussianBlur(dst, (3, 3), 0.9).clip(0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(path_save, imgnameA.replace("_A", str_)), dst + 2)
    #return dst
def unshiftmask(img_path,path_save,str_):
    # 使用高斯模糊平滑图像
    img = cv2.imread(img_path, 0)
    blur = cv2.GaussianBlur(img, (5, 5), 0)

    # 使用原始图像减去模糊后的图像，得到高频部分
    high_freq = cv2.subtract(img, blur)

    # 将高频部分加到原始图像上
    sharpened = cv2.addWeighted(img, 1.4, high_freq, -0.5, 0)
    cv2.imwrite(os.path.join(path_save, imgnameA.replace("_A", str_)),sharpened)
# kernel_5_2 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

threads = []

for pathA in paths_A:
    imgnameA = pathA.split("\\")[-1].replace(pathA.split("\\")[-1].split(".")[0], "R001C001_A")
    cycname = pathA.split("\\")[-2]
    print(cycname, " ", imgnameA)
    path_save = r"E:\data\highDensity\dense0.6\R001C001_unshapedmask2-1\Lane01\{}".format(cycname)
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    # cv2.imwrite(os.path.join(path_save, imgnameA), dstA + 2)
    # cv2.imwrite(os.path.join(path_save, imgnameA.replace("_A", "_C")), dstC + 2)
    # cv2.imwrite(os.path.join(path_save, imgnameA.replace("_A", "_G")), dstG + 2)
    # cv2.imwrite(os.path.join(path_save, imgnameA.replace("_A", "_T")), dstT + 2)

    # create threads for each channel
    threadA = threading.Thread(target=unshiftmask, args=(pathA,path_save,"_A"))
    threadC = threading.Thread(target=unshiftmask, args=(pathA.replace("R001C001_A","R001C001_C"),path_save,"_C"))
    threadG = threading.Thread(target=unshiftmask, args=(pathA.replace("R001C001_A", "R001C001_G"),path_save,"_G"))
    threadT = threading.Thread(target=unshiftmask, args=(pathA.replace("R001C001_A", "R001C001_T"),path_save,"_T"))

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




