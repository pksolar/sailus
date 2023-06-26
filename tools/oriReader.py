#!/usr/bin/env python3

###### Import Modules
import sys, os
import numpy as np
import struct
import matplotlib
import matplotlib.pyplot as plt
from itertools import combinations
import random
np.set_printoptions(precision=4, suppress=True) ## for test only

class OriginReader(object):
    INT_LEN = FLOAT_LEN = 4 # float and sign are 4 bytes
    UNSIGHED_SHORT_LEN = 2  # unsigned len are 2bytes
    FOV_CR_LEN = 3
    CHAN_NUM = 4
    
    def __init__(self, oriFile=None):
        self.clear()

        if oriFile:
            self.load(oriFile)

    def clear(self):
        ### clear old origin data
        self.dnbNum = 0
        self.realCyc = 0
        self.totalCyc = 0
        self.intSet = {}
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
            self.dnbNum = struct.unpack("i", fh.read(self.INT_LEN))[0]
            self.totalCyc = struct.unpack("i", fh.read(self.INT_LEN))[0]

    def dumpAll(self, path, compress = 1):
        ## dump all cycle into path
        for cycle in range(self.realcyc):
            self.dump(cycle + 1, path, compress)
    
    def dump(self, cycle, path, compress = 1):
        ## dump give cycle into path
        if not os.path.exists(path):
            os.makedirs(path)
        outfile = os.path.join(path, "%s_%03d_%s.txt" %(self.fov, cycle, os.path.splitext(os.path.basename(self.oriFile))[1][1:]))
            
        self.loadCyc(cycle, compress)
        np.savetxt(outfile, self.intArray.T, fmt='%.3f')

    def loadCyc(self, cycle, compress = 1):
        if cycle <= self.realcyc:
            with open(self.filename, 'rb', 10000000) as fh:
                #### Cross hearder
                fh.seek(3 * self.INT_LEN)
                '''self.dnbNum = struct.unpack("i", fh.read(self.INT_LEN))[0]
                self.realcyc = struct.unpack("i", fh.read(self.INT_LEN))[0]
                self.totalCyc = struct.unpack("i", fh.read(self.INT_LEN))[0]'''

                #### Skip the previous cycles
                if compress:
                    types_len = self.UNSIGHED_SHORT_LEN
                    types = np.float16
                else:
                    types_len = self.FLOAT_LEN
                    types = np.float32

                #### Skip the previous cycles
                fh.seek((cycle - 1) * self.CHAN_NUM * self.dnbNum * types_len, 1)

                self.intArray =  np.empty((self.CHAN_NUM, self.dnbNum),dtype = types)
                for channel in range(self.CHAN_NUM):
                    self.intArray[channel] = np.fromfile(fh, dtype=types, count=self.dnbNum)
                return self.intArray
                
        else:
            pass

def main():
    import argparse
    ArgParser = argparse.ArgumentParser()

    (para, args) = ArgParser.parse_known_args()
    
    if len(args) != 2:
        ArgParser.print_help()
        print(args)
        print(sys.stderr, "\nERROR: The parameters number is not correct!")
        sys.exit(1)
    else:
        (oriFile, cyc, ) = args
    ori = OriginReader(oriFile);
    int_array = ori.loadCyc(int(cyc))
    sampleSet = random.sample(list(range(np.shape(int_array)[1])), 10000)
    sum_cycle = []
    point_sum = []
    for i in range(1, 38):
        int_array = ori.loadCyc(int(i))
        print(len(int_array[0]))
        '''test= int_array[:, sampleSet].T
        point_sum.append(np.sum(test, 1))
        sum_cycle.append(np.sum(int_array[:, sampleSet]))'''
        #int_array2 = ori.loadCyc(int(cyc) + 4)
        '''plt.hist(np.max(int_array, axis = 0),bins=10,color='red',histtype='bar',rwidth=0.97, range=(0, float(bigX)))
        plt.ylim(0, int(bigY))
        #plt.hist2d(np.max(int_array, axis = 0)*100, np.max(int_array2, axis = 0)*100, bins=65, range=[[0, 200], [0, 200]])
        plt.show()'''
    np.savetxt("CycTotal_intent.csv", np.array(sum_cycle), delimiter = ",")
    np.savetxt("CycTotal_Point.csv", np.array(point_sum), delimiter = ",")

if __name__ == "__main__":
    main()


            
            
