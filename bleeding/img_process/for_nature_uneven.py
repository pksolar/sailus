import numpy as np
import cv2
import glob
import os
import math
#E:\code\python_PK\callbase\datasets\highdens_22_ori\Lane01\*\R001C001_A.tif




def unevenLightCompensate(gray, blockSizelist):
    dstList=[]
    for blockSize in blockSizelist:
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

        # max_value_ = np.max(blockImage)  # 最亮的9.741262。
        # print("biggest:",max_value_)
        # sorted_arr = np.sort(blockImage,0)
        # sorted_arr = np.sort(sorted_arr, 1)
        # max_value = sorted_arr[-1][-2]
        max_value = np.mean(blockImage)

        # # 输出第二大的数
        # print("second:",sorted_arr[-1][-2])
        # b = max_value / blockImage  # 系数
        aa,bb = blockImage.shape
        b = np.zeros([aa,bb])
        for i in range(aa):
            for j in range(bb):
                b[i,j] = math.log(max_value,blockImage[i,j]) #计算了这个尺度的指数矩阵。
        # if blockSize < 150:
        #     index = 0.9
        # else:
        #     index = 0.9
        blockImage2 = cv2.resize(b, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_CUBIC) * 0.8 #扩大这个尺度的指数矩阵。
        blockImage2[blockImage2<1 ]= 1
        blockImage2[blockImage2 > 1.8] = 1.8
        # print("max:",np.max(blockImage2))
        # print("max:", )
        gray_c = gray.copy()
        gray2 = gray_c.astype(np.float32)
        dst = gray2 ** blockImage2 #这个尺度的变换结果
        #dst_c = dst.copy()
        dstList.append(dst.copy())
    print("list长度：",len(dstList))
    dst = dstList[0]
    for i in range(len(dstList)-1):
        big = dstList[i+1].copy()
        big[big>255]=0
        dst = np.maximum(dst,big)
    return dst

img = cv2.imread("nature.tif",0)
cv2.imwrite("nature_gray.tif",img)
blockSizelist = [512]
dst =  unevenLightCompensate(img,blockSizelist).clip(0,255).astype(np.uint8)
cv2.imwrite("dst_nature.tif",dst)