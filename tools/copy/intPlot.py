#!/usr/bin/env python3

###### Import Modules
import sys, os
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
from itertools import combinations
from oriReader import OriginReader

prog_version = "0.1.0"
###### Usage
usage = '''
     version %s
     Usage: %s <intFile> <outPutDir> >STDOUT
''' % (prog_version,  os.path.basename(sys.argv[0]))

CHN = "ACGT"

class CrossTalkPlot(object):
    """docstring for CrossTalkPlot"""
    def __init__(self, intObj=None):
        # super(CrossTalkPlot, self).__init__()
        self.bins = 10
        self.vmaxF = 0.98
        self.vminF = 0.001
        self.sidx = []
        
        if isinstance(intObj, str):
            ## init from ori file
            self.intObj = OriginReader(intObj)
        elif intObj:
            ## init with OriginReader object
            self.intObj = intObj
        else:
            ## init with nothing
            pass

    def _sampling(self, intensity, size):
        ## Sampling display
        dnbNum = len(intensity[0])
        if size > dnbNum:
            samplingNum = dnbNum
        elif size <= 1:
            samplingNum = round(dnbNum*size)
        else:
            samplingNum = size

        dnbTotalArr = range(0, dnbNum)
        samplingArr = random.sample(dnbTotalArr, samplingNum)

        intensity = intensity.T
        intensity = intensity[samplingArr].T
        return intensity

    def plotAll(self, path, readSet = np.array([]), compress = 1, sampling = 200000, maskInvalid = True, addCyc = 0):
        ## Draw and save images at all cycles
        for cyc in range(self.intObj.realcyc):
            savePath = os.path.join(path, "crossPlot_%s_%003d.png" %(self.intObj.fov, cyc + 1 + addCyc))
            self.plot(cyc+1, readSet, compress, sampling, maskInvalid, False, savePath)
        
    def plot(self, cycle, readSet = np.array([]), compress = 1, sampling = 200000, maskInvalid = True, show = True, save = None, addCyc = 0):
        ## Draw crosstalk for the cycle
        intensity = self.intObj.loadCyc(cycle, compress)
        
        channels = list(combinations(['0', '1', '2', '3'], 2))
        pairNum = len(channels)

        ## fig size
        if pairNum == 1:
            col = 1
        elif pairNum <= 4:
            col = 2
        else:
            col = 3
        row = (len(channels) - 1) // col + 1
        unit = 10 - col * 2
        width = col * unit
        height = row * unit
        fig, subPlots = plt.subplots(nrows=row, ncols=col, figsize=(width, height))

        ## title
        titleText = "Crosstalk Plot"
        try:
            titleText = "Crosstalk Plot of %s - %003d" % (self.intObj.fov, cycle + addCyc)
        except KeyError:
            titleText = "Crosstalk Plot"
        fig.suptitle(titleText, fontsize=15, weight=560)

        ##  Sample the intensity
        if len(readSet) > 0:
            sampling = 1
            print(np.shape(intensity))
            intensity = intensity.take([readSet],1)
            intensity = np.squeeze(intensity)
            print(np.shape(intensity))
            '''intensity = intensity.T
            intensity = intensity[readSet].T'''
        intensity = self._sampling(intensity, sampling)

        ## remove invalid values
        if maskInvalid:
            mask = np.ones(intensity[0].shape, dtype=bool)
            for v in intensity:
                mask[np.where(v>=np.finfo(np.float16).max)] = False
            intensity = intensity.T
            intensity = intensity[mask].T

        ## max and min
        maxList = []
        minList = []
        for arr in intensity:
            maxIdx = int(self.vmaxF*len(arr))
            minIdx = int(self.vminF*len(arr))
            maxList.append(arr[np.argpartition(arr, maxIdx)[maxIdx]])
            minList.append(arr[np.argpartition(arr, minIdx)[minIdx]])
        vmax = max(maxList)
        vmin = min(minList)
        self.bin = int(vmax - vmin)
        
        tc = tr = 0
        for i,ch in enumerate(channels):
            subPlot = subPlots[tr][tc]
            subPlot.hist2d(intensity[int(ch[0])], intensity[int(ch[1])], bins=65, norm=matplotlib.colors.LogNorm(), density=True, cmap="jet", range=[[vmin, vmax], [vmin, vmax]]) ## normed=True, density=True
            #subPlot.hist2d(intensity[int(ch[0])], intensity[int(ch[1])], bins=65, norm=matplotlib.colors.LogNorm(), normed=True, cmap="jet", range=[[vmin, vmax], [vmin, vmax]]) ## normed=True, density=True
            subPlot.set_title("%s-%s" % (CHN[int(ch[0])], CHN[int(ch[1])]))
            subPlot.axis([vmin, vmax, vmin, vmax])
            tc += 1
            if tc == col:
                tc = 0
                tr += 1
        #plt.gca().set_aspect('equal', adjustable='box')
        if save:
            plt.savefig(save, dpi=250)

        if show:
            plt.show()

        plt.close()

def main():
    ######################### Phrase parameters #########################
    import argparse
    ArgParser = argparse.ArgumentParser(usage = usage)

    (para, args) = ArgParser.parse_known_args()

    if len(args) != 2:
        ArgParser.print_help()
        print(args)
        print >>sys.stderr, "\nERROR: The parameters number is not correct!"
        sys.exit(1)
    else:
        (intFile, outPut, ) = args
    ############################# Main Body #############################
    ori = OriginReader(intFile);
    plot = CrossTalkPlot(ori)
    plot.plotAll(outPut, np.array([]))
    #plot.plot(10)

if __name__ == "__main__":
    main()


            
            
