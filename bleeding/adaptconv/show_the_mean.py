import cv2
import glob
import numpy as np
"""
统计 100 个cycle ，图像的灰度均值。各个通道的灰度均值。

"""
lista =[]
listc = []
listg = []
listt = []
listtotal = []
list_acgt_mean=[]
name = "ori"
paths = glob.glob("E:\code\python_PK\callbase\datasets\highDens\Image\Lane01\*\R001C001_A.tif")
for pathA in paths:
    num = pathA.split("\\")[-2]
    print(num)

    imgA = cv2.imread(pathA, 0)


    imgC = cv2.imread(pathA.replace("_A", "_C"), 0)


    imgG = cv2.imread(pathA.replace("_A", "_G"), 0)


    imgT = cv2.imread(pathA.replace("_A", "_T"), 0)


    a = np.mean(imgA)
    c = np.mean(imgC)
    g = np.mean(imgG)
    t = np.mean(imgT)

    total = (a+c+g+t)/4

    lista.append(a)
    listc.append(c)
    listg.append(g)
    listt.append(t)
    listtotal.append(total)





list_acgt_mean=[np.mean(np.array(lista)),
                    np.mean(np.array(listc)),
                    np.mean(np.array(listg)),
                    np.mean(np.array(listt))]

np.savetxt("{}_a_mean.txt".format(name),lista,fmt='%.4f')
np.savetxt("{}_c_mean.txt".format(name),listc,fmt='%.4f')
np.savetxt("{}_g_mean.txt".format(name), listg,fmt='%.4f')
np.savetxt("{}_t_mean.txt".format(name), listt,fmt='%.4f')
np.savetxt("{}_total_mean.txt".format(name), listtotal,fmt='%.4f')
np.savetxt("{}_acgt_mean.txt".format(name), list_acgt_mean,fmt='%.4f')