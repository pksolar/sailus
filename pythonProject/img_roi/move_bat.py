import cv2
import numpy as np
import glob
import os
def move():
    img_temp = np.zeros([2160, 512], np.uint8)
    img_temp1 = img_temp.copy()
    img = cv2.imread(r"E:\code\python_PK\pythonProject\img_roi\s0\Lane01\\R001C001_A.tif", 0)
    img_temp1 = img[:, 0:512].copy()
    img[:, 0:512] = 1
    img[:, 3 * 512:4 * 512] = img_temp1

for k in range(11):
    i = 3
    j = 0
    img_temp = np.zeros([2160, 512], np.uint8)
    if k < 10:
        str_k = '0' + str(k)
    else:
        str_k = str(k)
    file_name1 = r"E:\code\python_PK\pythonProject\img_roi\s0\Lane01\Cyc0{}".format(str_k)
    file_name2 = glob.glob(file_name1 + "\*.tif")
    for file in file_name2:
        print(file)
        cycle_name = file.split("\\")[-2]
        image_name = file.split("\\")[-1]
        blockSize = 16
        img = cv2.imread(file, 0)

        img_temp1 = img[:, 0:512].copy()
        img[:, 0:512] = 1
        img[:, 3 * 512:4 * 512] = img_temp1

        print(os.path.exists("E:\code\python_PK\pythonProject\img_roi\s_move0\Lane01/" + cycle_name))
        if not os.path.exists("E:\code\python_PK\pythonProject\img_roi\s_move0\Lane01/" + cycle_name):
            os.mkdir("E:\code\python_PK\pythonProject\img_roi\s_move0\Lane01/" + cycle_name)
        save_name = "E:\code\python_PK\pythonProject\img_roi\s_move0\Lane01/" + cycle_name + "/" + image_name
        cv2.imwrite(save_name, img)
        j = j + 1
        print(j)
        if j > 11:
            j = 0
            break

