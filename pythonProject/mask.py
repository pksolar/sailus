import cv2
import numpy as np
import glob
import os

if __name__ == '__main__':

    i = 3
    j = 0

    for k in range(11):
        if k < 10:
            str_k = '0'+str(k)
        else:
            str_k = str(k)
        file_name1 = r"image_ori\Lane01\Cyc0{}".format(str_k)
        file_name2 = glob.glob(file_name1+"\*.tif")
        for file in file_name2:
            #print(file)
            cycle_name = file.split("\\")[-2]
            image_name = file.split("\\")[-1]
            blockSize = 256

            img = cv2.imread(file,0)
            #阈值分割：
            ret,thresh1 = cv2.threshold(img,12,255,cv2.THRESH_BINARY)
            img_and = cv2.bitwise_and(img,thresh1)


            cv2.imshow("thresh",thresh1)
            cv2.imshow("image_ori", img)
            cv2.imshow("img_and",img_and)
            cv2.imwrite("mask_img/img_and.tif",img_and)
            cv2.imwrite("mask_img/img.tif", img)
            cv2.waitKey(0)




            #dst = unevenLightCompensate(img, blockSize)

            #result = np.concatenate([img, dst], axis=1)
            name = "4"
            # cv2.imshow('image/result'.format(ele), result)
            print(os.path.exists("image_{}/Lane01/".format(name)+cycle_name))
            if not os.path.exists("image_{}/Lane01/".format(name)+cycle_name):
                os.mkdir("image_{}/Lane01/".format(name)+cycle_name)
            save_name = "image_{}/Lane01/".format(name)+cycle_name+"/"+image_name
            cv2.imwrite(save_name,dst)
            j = j + 1
            print(j)
            if j > 11:
                j  = 0
                break