#!/usr/bin/env python3
import sys, os
import numpy
import numpy as np
import glob
# import matplotlib.pyplot as plt
# import json
#
# from oriReader import OriginReader
# from calReader import CallReader

intFiexl = ".origin" #".origin"
prog_version = "0.1.0"
###### Usage
usage = '''
     version %s
     Usage: %s <posPath><intPath><samPath><outputPath> >STDOUT
''' % (prog_version,  os.path.basename(sys.argv[0]))

disThresh = 3;
imageHeigth = round(2700)
imageWidth = round(5120)

baseTrans = {"A" : 1, "C" : 2, "G" : 3, "T" : 4, "N" : 5}

def getPos(PosPath, intPath, samPath, outputPath):
    ### get coordinate files

    for posFile in glob.glob(os.path.join(PosPath, "R[0-9][0-9][0-9]C[0-9][0-9][0-9].temp")):
        
        fileName = os.path.basename(posFile)
        posSet = np.loadtxt(posFile, skiprows=2)

        fov = os.path.splitext(fileName)[0]
        # intFile = os.path.join(intPath, fov+intFiexl)
        callFile = os.path.join(samPath, fov+"correctACGT_False.txt")
        noMap = os.path.join(samPath, fov+"noMap.txt")

        if  os.path.exists(callFile) and os.path.exists(noMap):
            callResultOri = []
            # intObj = OriginReader(intFile);
            noMap = np.loadtxt(noMap)
            #callObj = CallReader(callFile)

            with open(callFile, 'r') as tf:
                for line in tf:
                    callResultOri.append(line)
            print("total cyc:", len(line)-1)
            for cyc in range(len(line)-1):
                print("cyc:",cyc)
                if not os.path.exists(os.path.join(outputPath, "Cyc%03d" % (cyc + 1))):
                    os.mkdir(os.path.join(outputPath, "Cyc%03d" % (cyc + 1)))
                if not os.path.exists(os.path.join(outputPath, "Cyc%03d" % (cyc + 1), "label")):
                    os.mkdir(os.path.join(outputPath, "Cyc%03d" % (cyc + 1), "label"))
                marker = np.zeros((imageHeigth, imageWidth), numpy.short)
                callResult = np.zeros((imageHeigth, imageWidth), numpy.ubyte)
                
                # intensity = intObj.loadCyc(cyc+1, 1)
                nums = 0
                errrNum = 0
                for pos in posSet:
                    int_x = round(pos[0])
                    int_y = round(pos[1])
                    if nums not in noMap: #nums 是readID
                        marker[int_y, int_x] = 1
                    else:
                        marker[int_y, int_x] = -1
                    # print("nums:",nums)
                    # print("cyc:", cyc)
                    callResult[int_y, int_x] = baseTrans[callResultOri[nums][cyc]] #callObj.calResult[nums]
                    nums = nums + 1
                np.save(os.path.join(outputPath, "Cyc%03d" % (cyc + 1), "label", "%s_label.npy" % (fov)),callResult)
                print("label done")

            np.save(os.path.join(outputPath, "%s_mask.npy" % (fov)), marker)
        else:
            print("%s or %s is not exist!" % (intFile, callFile))



def main():
    ######################### Phrase parameters #########################
    import argparse
    ArgParser = argparse.ArgumentParser(usage = usage)

    (para, args) = ArgParser.parse_known_args()

    if len(args) != 4:
        ArgParser.print_help()
        print >>sys.stderr, "\nERROR: The parameters number is not correct!"
        sys.exit(1)
    else:
        (PosPath, intPath, samPath, outputPath, ) = args
    os.makedirs(outputPath, exist_ok=True)
    getPos(PosPath, intPath, samPath, outputPath)

if __name__ == "__main__":
    main()

    

        
        
