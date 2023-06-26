import cv2
import numpy as np
import os
import glob
from calReader import CallReader
"""
已经知道了模板的id，读取每个cycle，每个channel的id，从原图上提取它们的intensity，得到每个通道所有的值，然后计算均值。
通过插值
"""

channl = {'A':0, 'C':1, 'G':2, 'T':3}#"ACGT"
OulierSuffix = "Outlier_Id.txt"
coordFIleCycChanSuffix = "_my.temp"
beforeLabelSuffix = "_beforeLabel.txt"
ImagePath = r"E:\code\python_PK\callbase\datasets\highDens\Image\Lane01"
outPath = r"E:\code\python_PK\callbase\datasets\highDens\result_outcorrd\Lane01\sfile\nearReadSet"
callPath = r"E:\code\python_PK\callbase\datasets\highDens\result_outcorrd\Lane01\sfile"
CycEnd = 100
coordDirCycChan = r"E:\code\python_PK\callbase\datasets\highDens\result_outcorrd\Lane01\sfile"


def getSubpixelInten(dst_x, dst_y, img):
    #### 插值法获得坐标亮度
    src_h, src_w = img.shape  # 原图片的高、宽、通道数

    # 源图像和目标图像几何中心的对齐
    # src_x = (dst_x + 0.5) * srcWidth/dstWidth - 0.5
    # src_y = (dst_y + 0.5) * srcHeight/dstHeight - 0.5
    src_x = (dst_x + 0.5) - 0.5
    src_y = (dst_y + 0.5) - 0.5

    # 计算在源图上四个近邻点的位置
    src_x0 = int(np.floor(src_x))
    src_y0 = int(np.floor(src_y))
    src_x1 = min(src_x0 + 1, src_w - 1)
    src_y1 = min(src_y0 + 1, src_h - 1)

    # 双线性插值
    temp0 = (src_x1 - src_x) * img[src_y0, src_x0] + (src_x - src_x0) * img[src_y0, src_x1]
    temp1 = (src_x1 - src_x) * img[src_y1, src_x0] + (src_x - src_x0) * img[src_y1, src_x1]
    subpixel_value = (src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1

    return subpixel_value

def getLableIntent(ImagePath, outPath, callPath, CycEnd, coordDirCycChan):
    ### get intensity for Outlier
    ### 1, load readId of Outlier;
    ### 2, get base of Outlier in each Cyc
    ### 3, load coord of Outlier in each Cyc and Chan
    ### 4, get Intent

    chanType = channl.keys()

    for callFile in glob.glob(os.path.join(callPath, "*.call")):
        print(callFile)
        FOV = os.path.splitext(os.path.basename(callFile))[0]
        if os.path.exists(os.path.join(outPath, FOV + OulierSuffix)):
            print(os.path.join(outPath, FOV + OulierSuffix))
            OulierRead = np.loadtxt(os.path.join(outPath, FOV + OulierSuffix)).T
            OutlierReadId = OulierRead[0].astype(np.int)

            callObj = CallReader(callFile)
            for cyc in range(CycEnd):
                print(cyc)
                callObj.loadCyc(cyc + 1)
                callBase = callObj.calResult[OutlierReadId]
                callScore = callObj.scoreResult[OutlierReadId] #q值

                intentRecord = [[], [], [], []] ### intA, intC, intG, intT, callBaes, coordX, coordY
                for ch in chanType:
                    #intentRecord[channl[ch]] = []
                    coordFileCycChanFile = os.path.join(coordDirCycChan, "%s_%d_%s" % (FOV, cyc + 1, ch) + coordFIleCycChanSuffix)
                    coordFileCycChan = np.loadtxt(coordFileCycChanFile)
                    Image = cv2.imread(os.path.join(ImagePath, "Cyc%03d" % (cyc + 1), "%s_%s.tif" % (FOV, ch)), -1)
                    for pointNum in range(len(OutlierReadId)):
                        if callScore[pointNum] > 25:
                            readId = OutlierReadId[pointNum]
                            x_Temp = coordFileCycChan[readId][0]
                            y_Temp = coordFileCycChan[readId][1]

                            intentRecord[channl[ch]].append(getSubpixelInten(x_Temp, y_Temp, Image))
                    intentRecord[channl[ch]] = np.array(intentRecord[channl[ch]])

                intentRecord.append([])
                for pointNum in range(len(OutlierReadId)):
                    if callScore[pointNum] > 25:
                        intentRecord[-1].append(callBase[pointNum])

                intentRecord.append([])
                for pointNum in range(len(OutlierReadId)):
                    if callScore[pointNum] > 25:
                        intentRecord[-1].append(OulierRead[1][pointNum])

                intentRecord.append([])
                for pointNum in range(len(OutlierReadId)):
                    if callScore[pointNum] > 25:
                        intentRecord[-1].append(OulierRead[2][pointNum])

                np.savetxt(os.path.join(outPath, "%s_%d" % (FOV, cyc + 1) + beforeLabelSuffix), np.array(intentRecord))

getLableIntent(ImagePath, outPath, callPath, CycEnd, coordDirCycChan)