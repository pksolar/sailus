import cv2
import numpy as np
import glob
import os

def msk_make():

    img_msk = np.zeros([2160, 4096], np.uint8)
    a, b = img_msk.shape #a = 2160 b = 4096
    num = 8
    msk_list = []
    print(a,b)
    step = int(b/num)
    for i in range(8):
        img_msk_copy = img_msk.copy()
        print((i+1)*step)
        img_msk_copy[:,i*step:(i+1)*step] = 255
        msk_list.append(img_msk_copy)
    return msk_list


def main():

    i = 3
    j = 0

    img_msk = np.zeros([2160, 4096], np.uint8)
    a, b = img_msk.shape  # a = 2160 b = 4096
    num = 8
    msk_list = []
    print(a, b)
    step = int(b / num)
    for i in range(8):
        img_msk_copy = img_msk.copy()
        print((i + 1) * step)
        img_msk_copy[:, i * step:(i + 1) * step] = 255
        msk_list.append(img_msk_copy)

    for k in range(11):
        if k < 10:
            str_k = '0'+str(k)
        else:
            str_k = str(k)
        file_name1 = r"..\\image_ori\Lane01\Cyc0{}".format(str_k)
        file_name2 = glob.glob(file_name1+"\*.tif")
        for file in file_name2:
            #print(file)
            cycle_name = file.split("\\")[-2]
            image_name = file.split("\\")[-1]
            img = cv2.imread(file,0)
            for i in range(8):
                msk = msk_list[i].copy() #msk
                img_and = cv2.bitwise_and(img, msk)
                msk[img_msk == 0] = 1
                msk[img_msk == 255] = 0
                img_and = img_and + img_msk
                if not os.path.exists("../img_roi/s{}/Lane01/".format(i)+cycle_name):
                    os.makedirs("../img_roi/s{}/Lane01/".format(i)+cycle_name)
                save_name = "../img_roi/s{}/Lane01/".format(i) + cycle_name + "/" + image_name

                img_temp1 = img_and[:, i * step:(i + 1) * step].copy()
                img_and[:, i * step:(i + 1) * step] = 1
                img_and[:, 3 * 512:4 * 512] = img_temp1


                cv2.imwrite(save_name,img_and)
                # cv2.imshow("img_and",img_and)
                # cv2.waitKey(0)

            # name = "4"
            # # cv2.imshow('image/result'.format(ele), result)
            # print(os.path.exists("image_{}/Lane01/".format(name)+cycle_name))
            # if not os.path.exists("image_{}/Lane01/".format(name)+cycle_name):
            #     os.mkdir("image_{}/Lane01/".format(name)+cycle_name)
            # save_name = "image_{}/Lane01/".format(name)+cycle_name+"/"+image_name
            # #cv2.imwrite(save_name,dst)
            # j = j + 1
            # print(j)
            # if j > 11:
            #     j  = 0
            #     break
main()
