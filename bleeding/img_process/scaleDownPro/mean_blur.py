import glob

import cv2
import numpy as np
import os
paths_A = glob.glob(r"E:\data\resize_test\08_resize_ori_for_yichuan\Lane01\*\R001C001_A.tif")
#17_R1C78_resize_oriE:\data\resize_test\17_R1C78_resize1.25
import threading
"""

    08号机ori：88.75
  ori 
  kernel = np.array([[0.02134374, 0.08381951, 0.02134374],
                       [0.16381951, -1.41934703, 0.16381951],
                       [0.02134374, 0.16381951, 0.02134374]])  mapping  89.6
                       
                       kernel = np.array([[0.04134374, 0.32381951, 0.04134374],
                       [0.32381951, -2.51934703, 0.32381951],
                       [0.04134374, 0.32381951, 0.04134374]]) # mapping 89.91
                       
    -1.000000000000000000e+00 -8.110236220472440971e-01 -1.000000000000000000e+00
-8.110236220472440971e-01 3.886497064579256477e+00 -8.110236220472440971e-01
-1.000000000000000000e+00 -8.110236220472440971e-01 -1.000000000000000000e+00
      
      
      -5.118110236220472231e-01 -3.464566929133858775e-01 -5.118110236220472231e-01
-3.464566929133858775e-01 3.876712328767123239e+00 -3.464566929133858775e-01
-5.118110236220472231e-01 -3.464566929133858775e-01 -5.118110236220472231e-01                 84.09


-4.409448818897637734e-01 -5.275590551181101873e-01 -4.409448818897637734e-01
-5.275590551181101873e-01 3.045009784735812186e+00 -5.275590551181101873e-01
-4.409448818897637734e-01 -5.275590551181101873e-01 -4.409448818897637734e-01   
                       
                       
                       
                       -6.141732283464567121e-01 -2.677165354330708347e-01 -6.141732283464567121e-01
-2.677165354330708347e-01 5.559686888454011822e+00 -2.677165354330708347e-01
-6.141732283464567121e-01 -2.677165354330708347e-01 -6.141732283464567121e-01
"""
def makeStrToKernel(s):


    # str = "-3.543307086614173596e-01 -9.527559055118109965e-01 -3.543307086614173596e-01 -9.527559055118109965e-01 5.246575342465753522e+00 -9.527559055118109965e-01 -3.543307086614173596e-01 -9.527559055118109965e-01 -3.543307086614173596e-01"
    x0 = s.split(" ")[0]
    x1 = s.split(" ")[1]
    x2 = s.split(" ")[4]
    x0 = float(x0[:5])/(10 ** float(x0[-1]))
    x1 = float(x1[:5]) / (10 ** float(x1[-1]))
    x2 = float(x2[:5]) / (10 ** float(x2[-1]))

    kernel = np.array([[x0, x1, x0],
                       [x1, x2,x1],
                       [x0, x1, x0]])
    return kernel


def convolve(img_path,path_save,str_):
    s = "ddd"
    kernel = makeStrToKernel(s)


    # kernel = np.array([[-0.614,-0.2677,-0.614],
    #                    [-0.2677, 5.559, -0.2677],
    #                    [-0.614,-0.2677,-0.614]])
    img = cv2.imread(img_path, 0)
    dst = cv2.filter2D(img, -1, kernel).astype(np.uint8)
    # dst = cv2.GaussianBlur(dst, (3, 3), 0.9).clip(0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(path_save, imgnameA.replace("_A", str_)), dst)
    #return dst



threads = []

for pathA in paths_A:
    imgnameA = pathA.split("\\")[-1].replace(pathA.split("\\")[-1].split(".")[0], "R001C001_A")
    cycname = pathA.split("\\")[-2]
    print(cycname, " ", imgnameA)
    path_save = r"E:\code\python_PK\bleeding\img_process\scaleDownPro\Gauss08_5.59\Lane01\{}".format(cycname)
    if not os.path.exists(path_save):
        os.makedirs(path_save)

    # create threads for each channel
    threadA = threading.Thread(target=convolve, args=(pathA,path_save,"_A"))
    threadC = threading.Thread(target=convolve, args=(pathA.replace("R001C001_A","R001C001_C"),path_save,"_C"))
    threadG = threading.Thread(target=convolve, args=(pathA.replace("R001C001_A", "R001C001_G"),path_save,"_G"))
    threadT = threading.Thread(target=convolve, args=(pathA.replace("R001C001_A", "R001C001_T"),path_save,"_T"))

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




