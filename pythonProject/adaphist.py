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
def adapthist(img):
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))

    dst = clahe.apply(img)  # 将灰度图像和局部直方图相关联
    return dst

def unevenLightCompensate(gray, blockSize):

    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #average = np.mean(gray)

    rows_new = int(np.ceil(gray.shape[0] / blockSize))

    cols_new = int(np.ceil(gray.shape[1] / blockSize))

    blockImage = np.zeros((rows_new, cols_new), dtype=np.float32)

    for r in range(rows_new):

        for c in range(cols_new):

            rowmin = r * blockSize

            rowmax = (r + 1) * blockSize

            if (rowmax > gray.shape[0]):

                rowmax = gray.shape[0]

            colmin = c * blockSize

            colmax = (c + 1) * blockSize

            if (colmax > gray.shape[1]):

                colmax = gray.shape[1]

            imageROI = gray[rowmin:rowmax, colmin:colmax]

            temaver = np.mean(imageROI)

            blockImage[r, c] = temaver

    max_value = 10 #np.max(blockImage)  # 最亮的9.741262。

    b = max_value / blockImage  # 系数

    #blockImage = blockImage - average

    blockImage2 = cv2.resize(b, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_CUBIC) #0.4的时候有提升。

    gray2 = gray.astype(np.float32)

    dst = gray2 * blockImage2
    #dst[dst==0]=0

    dst = dst.astype(np.uint8)

    # dst = cv2.GaussianBlur(dst, (3, 3), 0)
    #统计灰度直方图，让背景全部为0.
    #back_pixel  = pix_gray(dst)[0] + 1
    #dst[dst<back_pixel] = 0


    #dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)

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
            blockSize = 256

            img = cv2.imread(file,0)
            kernel = np.ones((25, 25), np.uint8)
            opening = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)#开运算： 先腐蚀后膨胀
            #cv2.imwrite("back50.tif",opening)
            qian_open = img - opening


            #dst = unevenLightCompensate(qian_open, blockSize)
            dst = adapthist(img)

            #opening = cv2.cvtColor(opening, cv2.COLOR_BGR2GRAY)
            # opening = opening * 0.5
            opening = opening.astype(np.uint8)
            dst = dst + opening
            # cv2.imshow("dst",dst)
            # cv2.waitKey(0)
            #result = np.concatenate([img, dst], axis=1)

            # cv2.imshow('image/result'.format(ele), result)
            name_ = "adapthist"
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
