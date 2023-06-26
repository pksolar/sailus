import glob

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
                            baseTemp = callResultOri[readId][cyc]
                        except:
                            print(readId)
                            print(np.shape(callResultOri))

                        outTemp = []
                        outTemp.append(readId)
                        if baseTemp == 'A':
                            outTemp.extend(coordFovCycA[readId])
                            labelA.append(outTemp)
                        elif baseTemp == 'C':
                            outTemp.extend(coordFovCycC[readId])
                            labelC.append(outTemp)
                        elif baseTemp == 'G':
                            outTemp.extend(coordFovCycG[readId])
                            labelG.append(outTemp)
                        elif baseTemp == 'T':
                            outTemp.extend(coordFovCycT[readId])
                            labelT.append(outTemp)
                np.savetxt(os.path.join(outPath, "%s_%d_%c%s" % (fov, cyc + 1, 'A', otherPointCycChanCorrd)), np.array(labelA))
                np.savetxt(os.path.join(outPath, "%s_%d_%c%s" % (fov, cyc + 1, 'C', otherPointCycChanCorrd)), np.array(labelC))
                np.savetxt(os.path.join(outPath, "%s_%d_%c%s" % (fov, cyc + 1, 'G', otherPointCycChanCorrd)), np.array(labelG))
                np.savetxt(os.path.join(outPath, "%s_%d_%c%s" % (fov, cyc + 1, 'T', otherPointCycChanCorrd)), np.array(labelT))

