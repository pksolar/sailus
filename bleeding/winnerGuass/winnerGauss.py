import glob

import cv2
import numpy as np
import os



paths_A = glob.glob(r"E:\data\highDensity\dense0.6\R001C001\Lane01\*\R001C001_A.tif")
import threading
# kernel = np.ones((3,3),np.uint8)
# def convolve(img_path,path_save,str_):
#     img = cv2.imread(img_path, 0)
#     img = cv2.erode(img,kernel)
#     cv2.imwrite(os.path.join(path_save, imgnameA.replace("_A", str_).replace("jpg","tif")), img)
#     #return dst
def gauss(img_path, path_save, str_):

    img = cv2.imread(img_path, 0)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    #
    # # 膨胀处理
    # # img = cv2.dilate(img, kernel, iterations=1)
    dst = cv2.GaussianBlur(img, (3, 3), 2).clip(0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(path_save, imgnameA.replace("_A", str_)), dst)
    #return dst



def convolve(img_path, path_save, str_):
    kernel = np.array([[-0.75, -0.5, -0.75],
                       [-0.5, 5.25, -0.5],
                       [-0.75, -0.5, -0.75]])

    kernel = np.array([[-0.1, -0.2, -0.3, -0.2, -0.1],
                           [-0.1, -0.3, -0.5, -0.3, -0.1],
                           [-0.3, -0.5, 5.5, -0.5, -0.5],
                           [-0.1, -0.3, -0.5, -0.3, -0.1],
                           [-0.1, -0.2, -0.3, -0.2, -0.1]])

    img = cv2.imread(img_path, 0)
    dst = cv2.filter2D(img, -1, kernel).astype(float)
    dst = cv2.GaussianBlur(dst, (3, 3), 1.2).clip(0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(path_save, imgnameA.replace("_A", str_)), dst)
    #return dst

def gamma_(img_path,path_save,str_):
    gamma = 1.25
    img = cv2.imread(img_path, 0)
    corrected_image = ((img / 180.0) ** gamma) * 180.0
    corrected_image = corrected_image.astype("uint8")
    cv2.imwrite(os.path.join(path_save, imgnameA.replace("_A", str_)), corrected_image)


threads = []

for pathA in paths_A:
    imgnameA = pathA.split("\\")[-1].replace(pathA.split("\\")[-1].split(".")[0], "R001C001_A")
    cycname = pathA.split("\\")[-2]
    print(cycname, " ", imgnameA)
    path_save = r"E:\code\python_PK\bleeding\winnerGauss\gauss_\Lane01\{}".format(cycname)
    if not os.path.exists(path_save):
        os.makedirs(path_save)

    # create threads for each channel
    threadA = threading.Thread(target=gauss, args=(pathA, path_save,"_A"))
    threadC = threading.Thread(target=gauss, args=(pathA.replace("R001C001_A","R001C001_C"),path_save,"_C"))
    threadG = threading.Thread(target=gauss, args=(pathA.replace("R001C001_A", "R001C001_G"),path_save,"_G"))
    threadT = threading.Thread(target=gauss, args=(pathA.replace("R001C001_A", "R001C001_T"),path_save,"_T"))

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




