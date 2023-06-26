import glob

import cv2
import numpy as np
import os



paths_A = glob.glob(r"E:\code\python_PK\bleeding\img_process\4channel_process\08_reg_add_bac\Lane01\*\R001C001_A.jpg")
import threading

def convolve(img_path,path_save,str_):
    img = cv2.imread(img_path, 0)

    cv2.imwrite(os.path.join(path_save, imgnameA.replace("_A", str_).replace("jpg","tif")), img + 1)
    #return dst

# kernel_5_2 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

threads = []

for pathA in paths_A:
    imgnameA = pathA.split("\\")[-1].replace(pathA.split("\\")[-1].split(".")[0], "R001C001_A")
    cycname = pathA.split("\\")[-2]
    print(cycname, " ", imgnameA)
    path_save = r"E:\code\python_PK\bleeding\img_process\4channel_process\08_reg_add_bac\Lane01\{}".format(cycname)
    if not os.path.exists(path_save):
        os.makedirs(path_save)

    # create threads for each channel
    threadA = threading.Thread(target=convolve, args=(pathA, path_save,"_A"))
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




