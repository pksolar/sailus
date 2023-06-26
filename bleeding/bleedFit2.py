#!/usr/bin/env python3

###### Import Modules #####
import sys, os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import glob
import cv2
import xopen
import json
import re

from calReader import CallReader
from oriReader import OriginReader

prog_version = "0.1.0"
###### Usage  #####
usage = '''
     version %s
     author %s
     Usage: %s <samFile> <originDir> <midDir> <outDir>>STDOUT
''' % (prog_version,  "chenwei 20220511", os.path.basename(sys.argv[0]))

baseTrans = {"A" : 1, "C" : 2, "G" : 3, "T" : 4, "N" : 5}

###### explain  #####
"""
   PRINCIPLE : When the number of cycle is small, the intensity is mainly affected by two factors : bleeding, crosstalk

               so If we remove the crosstalk factor, the intensity of non base channels should be 0,

   STEP : 1) Confirm which bases (set W) are mapped

          2) Obtain the intensity of these bases (set W) after crosstalk correction ;

          3) Fit the bleeding factor to that the intensity of the non base channel is corrected to 0

   OUTPUT : 1) bleeding factor; 2)  Fitted loss results         
"""
#####  global variable   ####
transTable = str.maketrans('ACGT', 'TGCA')
transDict = {"A":"T", "C":"G", "G":"C", "T":"A"}
channl = {'A':0, 'C':1, 'G':2, 'T':3}#"ACGT"
threshDis = 4.0

OulierSuffix = "Outlier_Id.txt"
coordFIleCycChanSuffix = "_my.temp"
intfileSuffix = ".ChanInt"
beforeLabelSuffix = "_beforeLabel.txt"
puritySuffix = "_allPurity.txt"
otherPointCycChanCorrd = "_otherCycChanCoord.txt"

#####  Parameter setting   #####

#####  stastic right call base #####

def mdzToList(mdz):
    ''' Parse MD:Z string into a list of operations, where 0=match,
        1=read gap, 2=mismatch. '''
    md = mdz[5:]
    i = 0;
    ret = []  # list of (op, run, str) tuples
    while i < len(md):
        if md[i].isdigit():  # stretch of matches
            run = 0
            while i < len(md) and md[i].isdigit():
                run *= 10
                run += int(md[i])
                i += 1  # skip over digit
            if run > 0:
                ret.append([0, run, ""])
        elif md[i].isalpha():  # stretch of mismatches
            mmstr = ""
            while i < len(md) and md[i].isalpha():
                mmstr += md[i]
                i += 1
            assert len(mmstr) > 0
            ret.append([1, len(mmstr), mmstr])
        elif md[i] == "^":  # read gap
            i += 1  # skip over ^
            refstr = ""
            while i < len(md) and md[i].isalpha():
                refstr += md[i]
                i += 1  # skip over inserted character
            assert len(refstr) > 0
            ret.append([2, len(refstr), refstr])
        else:
            raise RuntimeError('Unexpected character in MD:Z: "%d"' % md[i])
    return ret


def cigarToList(cigar):
    ''' Parse CIGAR string into a list of CIGAR operations.  For more
        info on CIGAR operations, see SAM spec:
        http://samtools.sourceforge.net/SAMv1.pdf '''
    ret, i = [], 0
    op_map = {'M': 0,  # match or mismatch
              '=': 0,  # match
              'X': 0,  # mismatch
              'I': 1,  # insertion in read w/r/t reference
              'D': 2,  # deletion in read w/r/t reference
              'N': 3,  # long gap due e.g. to splice junction
              'S': 4,  # soft clipping due e.g. to local alignment
              'H': 5,  # hard clipping
              'P': 6}  # padding
    # Seems like = and X together are strictly more expressive than M.
    # Why not just have = and X and get rid of M?  Space efficiency,
    # mainly.  The titans discuss: http://www.biostars.org/p/17043/
    while i < len(cigar):
        run = 0
        while i < len(cigar) and cigar[i].isdigit():
            # parse one more digit of run length
            run *= 10
            run += int(cigar[i])
            i += 1
        assert i < len(cigar)
        # parse cigar operation
        op = cigar[i]
        i += 1
        assert op in op_map
        # append to result
        ret.append([op_map[op], run])
    return ret


def findVar(infoCol, mdzCol):
    ####### Example, CIGAR: 28M1I5M   MD:Z 28M1I5M
    ####### cigarList: [[0, 28], [1, 1], [0, 5]]
    ####### mdzList:   [[0, 32, ''], [1, 1, 'G']]
    cigarList = cigarToList(infoCol[5])
    if mdzCol >= len(infoCol) or not infoCol[mdzCol].startswith('MD:Z'):
        mdzCol = _findcol(infoCol, 'MD:Z')
    mdzList = mdzToList(infoCol[mdzCol])
    mdIdx = 0  ## the index position in mdzlist
    seqPos = 0
    varList = []
    for cig in cigarList:
        cigType, cigLen = cig
        try:
            assert cigType == 4 or cigType == 5 or mdIdx < len(mdzList)
        except AssertionError:
            return False
        if cigType == 0:  ## CIGAR is M, =, X
            cigLeft = cigLen
            while cigLeft > 0 and mdIdx < len(mdzList):
                mdType, mdLen, mdBase = mdzList[mdIdx]
                lenComb = min(cigLeft, mdLen)
                cigLeft -= lenComb
                assert mdType == 0 or mdType == 1  ## MDZ should be match or mismatch
                if mdType == 1:  ## MD:Z got a mismatch
                    assert len(mdBase) == lenComb
                    for pos in range(mdLen):
                        seqPos += 1
                        #### type, position on seq, ref base, reads base, 0 means mismatch
                        varList.append([0, seqPos, mdBase[pos], infoCol[9][seqPos - 1]])

                else:  ## MD:Z got a match
                    seqPos += lenComb

                if lenComb < mdLen:
                    assert mdType == 0
                    mdzList[mdIdx][1] -= lenComb
                else:
                    mdIdx += 1
        elif cigType == 1:  ## CIGAR is I
            ####### reads insert against to reference, 1 means insertion
            varList.append([1, seqPos + 1, '-' * cigLen, infoCol[9][seqPos:seqPos + cigLen]])
            seqPos += cigLen  ## CIGAR move forward, but not apper on MD:Z
        elif cigType == 2:  ## CIGAR is D
            mdType, mdLen, mdBase = mdzList[mdIdx]
            assert mdType == 2 and cigLen == mdLen and cigLen == len(mdBase)
            varList.append([2, seqPos, mdBase, '-' * mdLen])
            # seqPos += mdLen
            mdIdx += 1
        elif cigType in (3, 4, 5):  ## CIGAR is N, S, H
            seqPos += cigLen
            ######## TODO: check here whether appropriated for N
        else:
            raise RuntimeError("Unknown CIGAR string: %s" % cigType)
    return varList


def _findcol(infoCol, marker):
    for idx, c in enumerate(infoCol[11:]):
        if c.startswith(marker):
            return idx + 11
    raise RuntimeError("Can't find %s column in sam file." % marker)


def revCompSeq(seq):
    trans = seq.translate(transTable)
    return trans[::-1]


def reverseVar(varList, seqLen):
    if not varList:
        varList = []
    for v in varList:
        if v[0] == 0:  ## type is mismatch
            v[1] = seqLen - v[1] + len(v[2])
        elif v[0] == 1:  ## type is insertion
            v[1] = seqLen - (v[1] + len(v[2])) + 1
        else:  ## type is deletion
            v[1] = seqLen - v[1]
        v[2] = revCompSeq(v[2])
        v[3] = revCompSeq(v[3])
    return varList


def staticSam(samLine, mdzCol):
    ## static sam line
    info = samLine.split()

    ### samStat
    isReverse = int(info[1]) & 16
    isUnmap = int(info[1]) & 4
    seqLen = len(info[9])

    if isUnmap:
        return info[9]

    result = list("1" * seqLen)

    if int(info[1]) & 16:
        varList = reverseVar(findVar(info, mdzCol), seqLen)
    else:
        varList = findVar(info, mdzCol)

    if int(info[1]) & 16:
        resultTemp = revCompSeq(info[9])
    else:
        resultTemp = info[9]

    resultTemp = list(resultTemp)
    if len(varList) == 0:
        return "".join(resultTemp)

    for var in varList:
        if var[0] == 0:
            resultTemp[var[1] - 1] = var[2]

    return "".join(resultTemp)

def staFq(fastqFile):
    fov_num = {}
    with xopen.xopen(fastqFile, threads=0) as pf:
        for line in pf:
            ### read infor

            ### for xopen.xopen
            title = line
            read_seq = pf.readline()
            Links = pf.readline()
            Q_value = pf.readline()

            p = re.compile(r'R\d{3}C\d{3}')
            pres = p.findall(title);
            ### Parse the seq ID to process.
            if(len(pres) != 0):
                fov = pres[0]

                if fov not in fov_num.keys():
                    fov_num[fov] = 0

                fov_num[fov] += 1
            else:
                print(title)
    return fov_num

##### end stastic right read #####

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

def getOutlier(coordDirTotal, outPath):
    ### get Outlier for each FOV
    for coordFile in glob.glob(os.path.join(coordDirTotal, "*.temp")):
        if '_' not in os.path.basename(coordFile):
            print(coordFile)
            FOV = os.path.splitext(os.path.basename(coordFile))[0]
            peakT = np.loadtxt(coordFile, skiprows=2)
            peakT = peakT.astype(np.float32)
            peak = peakT.T
            nearReadSet = []  ### reacord Outlier readId
            for readId, peakTemp in enumerate(peakT):
                #### calculate each
                nearPos = np.where((peak[0] < peakTemp[0] + threshDis) & (peak[0] > peakTemp[0] - threshDis) &
                         (peak[1] < peakTemp[1] + threshDis) & (peak[1] > peakTemp[1] - threshDis))[0]
                if len(nearPos) == 1:
                    nearReadSet.append([readId, peakTemp[0], peakTemp[1]])
                    continue
                else:
                    difX = np.square(peak[0][nearPos] - peakTemp[0])
                    difY = np.square(peak[1][nearPos] - peakTemp[1])
                    disTemp = np.sqrt(difX + difY)  # np.sqrt(difX + difY)

                    nearNum = len(np.where(disTemp < threshDis)[0])

                    if nearNum == 1:
                        #print(np.sort(disTemp))
                        nearReadSet.append([readId, peakTemp[0], peakTemp[1]])
            #### out nearRead Id to file
            np.savetxt(os.path.join(outPath, FOV + OulierSuffix), np.array(nearReadSet))

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

def makePurity(outPath):
    ### 制作purity的标签
    for beforeLabelFile in glob.glob(os.path.join(outPath, "*" + beforeLabelSuffix)):
        beforeLabel = np.loadtxt(beforeLabelFile).T
        purity = [[], [], [], []]
        print(os.path.splitext(os.path.basename(beforeLabelFile))[0])
        FOV, cycNum = os.path.splitext(os.path.basename(beforeLabelFile))[0].split('_')[:2]
        for point in beforeLabel:
            callBase = int(point[4])
            purityTemp = point[callBase] / sum(point[:4])
            purity[callBase].append([purityTemp, point[5], point[6]])
        np.savetxt(os.path.join(outPath, "%s_%s_A" % (FOV, cycNum) + puritySuffix), purity[0])
        np.savetxt(os.path.join(outPath, "%s_%s_C" % (FOV, cycNum) + puritySuffix), purity[1])
        np.savetxt(os.path.join(outPath, "%s_%s_G" % (FOV, cycNum) + puritySuffix), purity[2])
        np.savetxt(os.path.join(outPath, "%s_%s_T" % (FOV, cycNum) + puritySuffix), purity[3])

def makeTrain(outPath, samFile, fastqFile):
    ### 获取每个read的正确base和坐标 ###
    ##  stastic  fastq file;
    fovNum = staFq(fastqFile)
    print(fovNum)
    fovNumRecord = {}
    fovString = {}
    fovMask = {}

    ###### Skip Read Group Line
    sam = open(samFile, 'r', 100000000, encoding='gb18030', errors='ignore')  #
    firstPos = 0
    firstLine = sam.readline()
    while firstLine.startswith('@'):
        firstPos = sam.tell()
        firstLine = sam.readline()
    info = firstLine.split()

    try:
        while int(info[1]) & 4:
            info = sam.readline().split()
    except IndexError as e:
        sam.close()
    sam.close()

    ## lookup uniq mapping marker and MD:Z line
    mdzCol = _findcol(info, 'MD:Z')

    sam_result = {}
    with open(samFile, "r") as sam:
        sam.seek(firstPos)
        for sam_line in sam:
            info = sam_line.split()
            p = re.compile(r'R\d{3}C\d{3}')
            pres = p.findall(info[0]);

            readId = info[0].split("_")[2]
            ### Parse the seq ID to process.

            if (len(pres) != 0):
                fov = pres[0]

            if fov not in fovNumRecord.keys():
                fovNumRecord[fov] = 0
                fovString[fov] = []
                fovString[fov] = [""] * fovNum[fov]
                fovMask[fov] = []

            # if (len(info[9]) == 150):
            samResult = staticSam(sam_line, mdzCol)
            if (sam_line.split()[1] != '4'):
                # print(sam_line.split()[1])
                for pos in range(len(samResult) - 1):
                    q_value = ord(info[10][pos]) - 33
            else:
                fovMask[fov].append(readId)

            fovString[fov][int(readId)] = samResult
            fovNumRecord[fov] += 1

            if fovNumRecord[fov] == fovNum[fov]:
                f = open(os.path.join(outPath, fov + "correctACGT.txt"), "w")
                lineNum = 0
                for line in fovString[fov]:
                    if len(line) < 5:
                        print(lineNum)
                    lineNum += 1
                    f.write(line + '\n')
                # print(lineNum)
                f.close()
                fovString.pop(fov)
                fovNumRecord.pop(fov)
                # np.savetxt(os.path.join(outPath, fov + "noMap.txt"), fovMask[fov])
                f = open(os.path.join(outPath, fov + "noMap.txt"), "w")
                for line in fovMask[fov]:
                    f.write(line + '\n')
                f.close()
                fovMask.pop(fov)

def otherPointLabel(coordDirTotal, outPath, CycEnd):
    ### 获取非孤立点的碱基序列
    for coordFile in glob.glob(os.path.join(coordDirTotal, "R[0-9][0-9][0-9]C[0-9][0-9][0-9].temp")):
        fov = os.path.splitext(os.path.basename(coordFile))[0]
        callResultOri = []
        realCallFile = os.path.join(outPath, fov + "correctACGT.txt")
        print(realCallFile)
        with open(realCallFile, 'r') as tf:
            for line in tf:
                callResultOri.append(line)
        print(len(callResultOri))

        noMap = np.loadtxt(os.path.join(outPath, fov+"noMap.txt"))
        if os.path.exists(realCallFile) and os.path.exists(os.path.join(outPath, fov+"noMap.txt")):
            for cyc in range(CycEnd):
                labelA = []
                labelC = []
                labelG = []
                labelT = []

                coordFovCycA = np.loadtxt(os.path.join(outPath, "%s_%d_%c%s" % (fov, cyc + 1, 'A', coordFIleCycChanSuffix)))
                coordFovCycC = np.loadtxt(os.path.join(outPath, "%s_%d_%c%s" % (fov, cyc + 1, 'C', coordFIleCycChanSuffix)))
                coordFovCycG = np.loadtxt(os.path.join(outPath, "%s_%d_%c%s" % (fov, cyc + 1, 'G', coordFIleCycChanSuffix)))
                coordFovCycT = np.loadtxt(os.path.join(outPath, "%s_%d_%c%s" % (fov, cyc + 1, 'T', coordFIleCycChanSuffix)))

                for readId in range(len(callResultOri)):
                    if readId not in noMap:
                        try:
                            baseTemp = baseTrans[callResultOri[readId][cyc]]
                        except:
                            print(readId)
                            print(np.shape(callResultOri))

                        outTempA = [readId, baseTemp]
                        outTempC = [readId, baseTemp]
                        outTempG = [readId, baseTemp]
                        outTempT = [readId, baseTemp]

                        outTempA.extend(coordFovCycA[readId])
                        labelA.append(outTempA)

                        outTempC.extend(coordFovCycC[readId])
                        labelC.append(outTempC)

                        outTempG.extend(coordFovCycG[readId])
                        labelG.append(outTempG)

                        outTempT.extend(coordFovCycT[readId])
                        labelT.append(outTempT)
                        #print(labelA)
                np.savetxt(os.path.join(outPath, "%s_%d_%c%s" % (fov, cyc + 1, 'A', otherPointCycChanCorrd)), np.array(labelA))
                np.savetxt(os.path.join(outPath, "%s_%d_%c%s" % (fov, cyc + 1, 'C', otherPointCycChanCorrd)), np.array(labelC))
                np.savetxt(os.path.join(outPath, "%s_%d_%c%s" % (fov, cyc + 1, 'G', otherPointCycChanCorrd)), np.array(labelG))
                np.savetxt(os.path.join(outPath, "%s_%d_%c%s" % (fov, cyc + 1, 'T', otherPointCycChanCorrd)), np.array(labelT))

def main():
    ######################### Phrase parameters #########################
    import argparse
    ArgParser = argparse.ArgumentParser(usage = usage)

    (para, args) = ArgParser.parse_known_args()

    if len(args) != 8:
        ArgParser.print_help()
        print >>sys.stderr, "\nERROR: The parameters number is not correct!"
        sys.exit(1)
    else:
        (ImagePath, TotalCoordPath, outputPath, callPath, CycEnd, coordDirCycChan, samFile, fastqFile) = args
    ############################# Main Body #############################
    
    #getOutlier(TotalCoordPath, outputPath)
    #getLableIntent(ImagePath, outputPath, callPath, int(CycEnd), coordDirCycChan)
    #makePurity(outputPath)
    #makeTrain(outputPath, samFile, fastqFile)
    otherPointLabel(TotalCoordPath, outputPath, int(CycEnd))
    #compareQObj.getCorrectRead()

if __name__ == "__main__":
    main()