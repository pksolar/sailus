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
        sorted_arr = np.sort(blockImage,0)
        sorted_arr = np.sort(sorted_arr, 1)
        max_value = sorted_arr[-1][-2]

        # # 输出第二大的数
        # print("second:",sorted_arr[-1][-2])
        b = max_value / blockImage  # 系数
        # aa,bb = blockImage.shape
        # b = np.zeros([aa,bb])
        # for i in range(aa):
        #     for j in range(bb):
        #         b[i,j] = math.log(max_value,blockImage[i,j]) #计算了这个尺度的指数矩阵。
        # if blockSize < 150:
        #     index = 0.9
        # else:
        #     index = 0.9
        blockImage2 = cv2.resize(b, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_CUBIC)  #扩大这个尺度的指数矩阵。
        # blockImage2[blockImage2<0.8] = 0.8
        # blockImage2[blockImage2 > 1.8] = 1.8
        # print("max:",np.max(blockImage2))
        # print("max:", )
        gray_c = gray.copy()
        gray2 = gray_c.astype(np.float32)
        dst = gray2 * blockImage2 #这个尺度的变换结果
        #dst_c = dst.copy()
        dstList.append(dst.copy())
    print("list长度：",len(dstList))
    dst = dstList[0]
    for i in range(len(dstList)-1):
        big = dstList[i+1].copy()
        big[big>255]=255
        dst = np.maximum(dst,big)
    return dst


def warpuneven(img):
    blockSizelist = [512, 256, 128]
    # img = cv2.imread(file, 0)
    kernel = np.ones((25, 25), np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)  # 开运算： 先腐蚀后膨胀
    # cv2.imwrite("back50.tif",opening)
    qian_open = img - opening
    dst = unevenLightCompensate(qian_open, blockSizelist)
    # opening = cv2.cvtColor(opening, cv2.COLOR_BGR2GRAY)
    # opening = opening * 0.5
    #opening = opening.astype(np.uint8)
    dst = dst + opening
    # g += np.sum(dst > 255)
    # print("大于255的个数：", g)
    dst[dst > 255] = 255
    dst = dst.astype(np.uint8)
    return  dst



paths_x = glob.glob(r"E:\data\resize_test\08_resize_ori\Lane01\*\R001C001_A.tif")
save_dir = "08_unevenLight_mul"



#1.35 5530,2916
i= 0
height = 2160#2700
width = 4096#5120



for path in paths_x:
    name = path.split("\\")[-2]
    imgname = path.split("\\")[-1][:8]
    aA = cv2.imread(path, 0)
    aA = warpuneven(aA)
    aC = cv2.imread(path.replace("_A", "_C"), 0)
    aC = warpuneven(aC)
    aG = cv2.imread(path.replace("_A", "_G"), 0)
    aG = warpuneven(aG)
    aT = cv2.imread(path.replace("_A", "_T"), 0)
    aT = warpuneven(aT)
    print(imgname)

    if not os.path.exists("{}/Lane01/{}".format(save_dir, name)):
        os.makedirs("{}/Lane01/{}".format(save_dir, name))
    cv2.imwrite("{}/Lane01/{}/{}_A.tif".format(save_dir, name,imgname), aA)
    cv2.imwrite("{}/Lane01/{}/{}_C.tif".format(save_dir, name,imgname), aC)
    cv2.imwrite("{}/Lane01/{}/{}_G.tif".format(save_dir, name,imgname), aG)
    cv2.imwrite("{}/Lane01/{}/{}_T.tif".format(save_dir, name,imgname), aT)
    i += 1
    print(i)
