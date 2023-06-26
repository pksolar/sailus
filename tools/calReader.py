#!/usr/bin/env python3

###### Import Modules
import sys, os
import numpy as np
import struct
import matplotlib
import matplotlib.pyplot as plt
from itertools import combinations


import time

prog_version = "0.1.0"

###### global Variable
INT_LEN = 4 # int and sign are 4 bytes
FLOAT_LEN = 4 # float and sign are 4 bytes
UNSIGHED_SHORT_LEN = 2  # unsigned len are 2bytes
CHAR_LEN = 1  # unsigned len are 2bytes
    
###### Usage
usage = '''
     version %s
     Usage: %s <callFile> >STDOUT
''' % (prog_version,  os.path.basename(sys.argv[0]))

class CallReader(object):
    INT_LEN = 4 # int and sign are 4 bytes
    FLOAT_LEN = 4 # float and sign are 4 bytes
    UNSIGHED_SHORT_LEN = 2  # unsigned len are 2bytes
    CHAR_LEN = 1  # unsigned len are 2bytes
    UNSIGHED_CHAR_LEN = 2
    FOV_CR_LEN = 3
    CHAN_NUM = 4
    
    def __init__(self, callFile=None):
        self.clear()

        if callFile:
            self.load(callFile)

    def clear(self):
        ### clear old origin data
        self.dnbNum = 0
        self.chanNum = 0
        self.CHN = ''
        self.realCyc = 0
        self.totalCyc = 0
        self.fov = ""
        self.size = 0

    def getFov(self, filename):
        ## get the Row and Column of OriFile
        nameLen = 2 * (self.FOV_CR_LEN + 1)
        base = os.path.basename(filename)
        if base[0] == "R" and base[self.FOV_CR_LEN + 1] == "C":
            return base[:nameLen]
        return ""

    def load(self, filename):
        ### load OriginFile get infor
        self.clear()
        self.filename = filename
        self.fov = self.getFov(filename)
        self.size = os.stat(filename).st_size
        self.oriFile = filename
        with open(filename, 'rb', 10000000) as fh:
            #### load hearder
            self.realcyc = struct.unpack("i", fh.read(self.INT_LEN))[0]
            self.chanNum = struct.unpack("i", fh.read(self.INT_LEN))[0]
            for chann in range(self.chanNum):
                self.CHN += str(struct.unpack("c", fh.read(self.CHAR_LEN))[0])[2]
            self.dnbNum = struct.unpack("i", fh.read(self.INT_LEN))[0]
            self.totalCyc = struct.unpack("i", fh.read(self.INT_LEN))[0]
            print(" self.dnbNum : %d   self.totalCyc : %d " % (self.dnbNum, self.totalCyc));

    def dumpAll(self, path, compress = 1):
        ## dump all cycle into path
        for cycle in range(self.realcyc):
            self.dump(cycle + 1, path, compress)
    
    def dump(self, cycle, path, compress = 1):
        ## dump give cycle into path
        if not os.path.exists(path):
            os.makedirs(path)
        outfile = os.path.join(path, "%s_%03d_base.txt" %(self.fov, cycle))
            
        self.loadCyc(cycle, compress)
        np.savetxt(outfile, self.calResult.T, fmt='%d')
        

    def loadCyc(self, cycle, compress = 1):
        if cycle <= self.realcyc + 1:
            with open(self.filename, 'rb', 10000000) as fh:
                #### Cross hearder
                fh.seek(4 * self.INT_LEN + self.chanNum * self.CHAR_LEN, 0)

                #### Skip the previous cycles
                types_len = self.CHAR_LEN
                if compress:
                    cycle_len = 2
                else:
                    cycle_len = 3

                #### Skip the previous cycles
                fh.seek((cycle - 1) * cycle_len * self.dnbNum * types_len, 1)

                self.calResult = np.empty((self.dnbNum),dtype = np.int)
                self.scoreResult = np.empty((self.dnbNum),dtype = np.int)
                self.purityResult = np.empty((self.dnbNum),dtype = np.int)

                ### old mode maybe is slow
                '''for dnbTemp in range(self.dnbNum):
                    ### charity: 8bit    score: 6bit     base :2bit
                    total = struct.unpack("H", fh.read(UNSIGHED_SHORT_LEN))[0]
                    #total = int.from_bytes(total, 'big')
                    callContent = int(total&3)#total >> 14
                    scoreContent = total >> 2
                    scoreContent = int(scoreContent&63)
                    purityContent = total >> 8
                    purityContent = int(purityContent&254)

                    self.calResult[dnbTemp] = callContent
                    self.scoreResult[dnbTemp] = scoreContent
                    self.purityResult[dnbTemp] = purityContent'''

                ### new mode maybe is fast
                ### charity: 8bit    score: 6bit     base :2bit
                '''total = struct.unpack("BB"*self.dnbNum, fh.read(UNSIGHED_SHORT_LEN*self.dnbNum))
                for dnbTemp in range(self.dnbNum):
                    baseScore = total[2*dnbTemp]
                    purityContent = total[2*dnbTemp + 1]
                    
                    callContent = baseScore&3
                    scoreContent = baseScore >> 2

                    self.calResult[dnbTemp] = callContent
                    self.scoreResult[dnbTemp] = scoreContent
                    self.purityResult[dnbTemp] = purityContent'''

                ### new mode2 maybe is Fastest
                ### charity: 8bit    score: 6bit     base :2bit
                total = struct.unpack("BB"*self.dnbNum, fh.read(UNSIGHED_SHORT_LEN*self.dnbNum))
                self.calResult = np.array([total[baseNum]&3 for baseNum in range(self.dnbNum)])
                self.scoreResult = np.array([total[baseNum] >> 2 for baseNum in range(self.dnbNum)])
                self.purityResult = np.array([total[baseNum] for baseNum in range(self.dnbNum)])
                
        else:
            pass

def main():
    ######################### Phrase parameters #########################
    import argparse
    ArgParser = argparse.ArgumentParser(usage = usage)

    (para, args) = ArgParser.parse_known_args()
    
    if len(args) != 2:
        ArgParser.print_help()
        print(args)
        print(sys.stderr, "\nERROR: The parameters number is not correct!")
        sys.exit(1)
    else:
        (callFile, outpath) = args
    ############################# Main Body #############################
    ori = CallReader(callFile);
    
    result_name = []
    fovName = os.path.splitext(os.path.basename(callFile))[0]
    
    t = time.time()
    for j in range(1):
        for i in range(1, ori.realcyc + 1):
            ori.loadCyc(i)
            scoreResult = ori.calResult
            print(scoreResult)
            '''plt.hist(scoreResult,bins=10,color='red',histtype='bar',rwidth=0.97, range=(0, 36))
            plt.savefig(os.path.join(outpath, "%s_Cyc%03d.png" % (fovName, i)), dpi = 250)'''
        
    print(time.time() - t)
    '''print(a)
    print(len(a))'''

if __name__ == "__main__":
    main()


            
            
