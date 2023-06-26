import cv2
import numpy as np
import glob
import os


# 直方图统计

def pix_gray(img_gray):
    h = img_gray.shape[0]
    w = img_gray.shape[1]

    gray_level = np.zeros(256)
    gray_level2 = np.zeros(256)

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            gray_level[img_gray[i, j]] += 1  # 统计灰度级为img_gray[i,j的个数
    a = np.where(gray_level == max(gray_level))

    return a

    # for i in range(1, 256):
    #     gray_level2[i] = gray_level2[i - 1] + gray_level[i]  # 统计灰度级小于img_gray[i,j]的个数

def gama_transfer(img, power1):

    img = 255*np.power(img/255,power1)
    img = np.around(img)
    img[img>255] = 255
    dst = img.astype(np.uint8)

    return dst

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
            blockSize = 1000

            img = cv2.imread(file,0)
            kernel = np.ones((25, 25), np.uint8)
            opening = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)#开运算： 先腐蚀后膨胀
            #cv2.imwrite("back50.tif",opening)
            qian_open = img - opening


            dst = gama_transfer(qian_open, 0.9)

            #opening = cv2.cvtColor(opening, cv2.COLOR_BGR2GRAY)
            # opening = opening * 0.5
            opening = opening.astype(np.uint8)
            dst = dst + opening
            # cv2.imshow("dst",dst)
            # cv2.waitKey(0)
            #result = np.concatenate([img, dst], axis=1)

            # cv2.imshow('image/result'.format(ele), result)
            name_ = "gamma0.9"
            print(os.path.exists("image_{}/Lane01/".format(name_)+cycle_name))
            if not os.path.exists("image_{}/Lane01/".format(name_)+cycle_name):
                os.makedirs("image_{}/Lane01/".format(name_)+cycle_name)
            save_name = "image_{}/Lane01/".format(name_)+cycle_name+"/"+image_name
            cv2.imwrite(save_name,dst)
            j = j + 1
            print(j)
            if j > 11:
                j  = 0
                break
