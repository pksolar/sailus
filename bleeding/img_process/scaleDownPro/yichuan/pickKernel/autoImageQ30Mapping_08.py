import glob

import cv2
import numpy as np
import os
import threading
import autoQ30mapping
"""
0.000000000000000000e+00 -6.165437730737638455e-02 0.000000000000000000e+00
-6.165437730737638455e-02 1.000000000000000000e+00 -6.165437730737638455e-02
0.000000000000000000e+00 -6.165437730737638455e-02 0.000000000000000000e+00
"""
#原始图像所在文件夹的A图：
paths_A = glob.glob(r"E:\data\highDensity\dense0.6\R1C1yichuan\Lane01\*\R001C001_A.tif")
# 保存结果的文件夹名
dirname = "pso_44_-0.643-6.69_30cycle"#-0.643-6.69
path_save = fr"E:\code\python_PK\bleeding\img_process\scaleDownPro\{dirname}\image\Lane01\\"
if not os.path.exists(path_save):
    os.makedirs(path_save)
#做mapping时指定的文件夹位置
rootdir = fr"E:\code\python_PK\bleeding\img_process\scaleDownPro\{dirname}\image\\"
resdir = "res30"
cycNum  = 30  # 50个cycle

def makeStrToKernel(s):

    """

    :param s: 输入的矩阵字符串
    :return: 转化为numpy矩阵 kernel
    """
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

def convolve(img_path,kernel, path_save, str_,imgnameA):
    # kernel = np.array([[0.1134374, 0.22381951, 0.1134374],
    #                    [0.22381951, -2.51934703, 0.22381951],
    #                    [0.1134374, 0.22381951, 0.1134374]])
    """
    写入卷积后的图像

    :param img_path: 原始图像路径
    :param kernel:
    :param path_save:
    :param str_:
    :param imgnameA: 原始图像名
    :return:
    """
    img = cv2.imread(img_path, 0)
    dst = cv2.filter2D(img, -1, kernel).astype(np.uint8)
    cv2.imwrite(os.path.join(path_save, imgnameA.replace("_A", str_)), dst)


def evaluate_kernel(rootdir,resdir,cycNum):
    #放入核参数，注意，空格。
    """
    [[-0.0629 -0.275  -0.0629]
        [-0.275   2.526  -0.275 ]
        [-0.0629 -0.275  -0.0629]]

    :param rootdir:
    :param resdir:
    :param cycNum:
    :return:


    -6.432911834154866337e-01 0.000000000000000000e+00 -6.432911834154866337e-01
0.000000000000000000e+00 5.694895379943655911e+00 0.000000000000000000e+00
-6.432911834154866337e-01 0.000000000000000000e+00 -6.432911834154866337e-01
    """


    params = [-0.643,0,5.69]
    #params = [0, 0,1]
    kernel = np.array([[params[0], params[1], params[0]],
                       [params[1], params[2], params[1]],
                       [params[0], params[1], params[0]]])
    # s = "-6.456104344089941272e-01 0.000000000000000000e+00 -6.456104344089941272e-01 0.000000000000000000e+00 5.701672352375958930e+00 0.000000000000000000e+00 -6.456104344089941272e-01 0.000000000000000000e+00 -6.456104344089941272e-01"
    # kernel = makeStrToKernel(s)



    #对图像采用多线程卷积操作：
    threads = []

    for pathA in paths_A:
        imgnameA = pathA.split("\\")[-1].replace(pathA.split("\\")[-1].split(".")[0], "R001C001_A")
        cycname = pathA.split("\\")[-2]
        print(cycname, " ", imgnameA)
        path_save = fr"E:\code\python_PK\bleeding\img_process\scaleDownPro\{dirname}\image\Lane01\{cycname}"
        if not os.path.exists(path_save):
            os.makedirs(path_save)

        # create threads for each channel
        threadA = threading.Thread(target=convolve, args=(pathA,kernel, path_save, "_A",imgnameA))
        threadC = threading.Thread(target=convolve, args=(pathA.replace("R001C001_A", "R001C001_C"),kernel, path_save, "_C",imgnameA))
        threadG = threading.Thread(target=convolve, args=(pathA.replace("R001C001_A", "R001C001_G"),kernel, path_save, "_G",imgnameA))
        threadT = threading.Thread(target=convolve, args=(pathA.replace("R001C001_A", "R001C001_T"),kernel, path_save, "_T",imgnameA))

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
    #图像做完，autoq30,automapping:

    mappingrate,mappedreads = autoQ30mapping.main(rootdir,resdir,cycleNum=cycNum)
    with open(rootdir+'\kernel_mapping.txt', 'w') as f:
        # f.write(s+"\n")
        f.write("mapping rate:"+ str(mappingrate))
    print( "mapping: "+str(mappingrate)+",mapped Reads: "+str(mappedreads))
    print("kernel:",kernel)



evaluate_kernel(rootdir,resdir,cycNum)