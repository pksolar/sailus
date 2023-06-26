import numpy as np
import glob
#切块：
# h = 2160
# w = 4096
# cropsize = 256
#
# row = np.floor(h/cropsize)
# col = np.floor(w/cropsize)
# for i in range(row):
#     for j in range(col):
#

pathes  = glob.glob(r"E:\code\python_PK\callbase\datasets\21\Res\Lane01\deepLearnData\*\intensity_norm\\R001C001_A.npy")
path_mask = r"E:\code\python_PK\callbase\datasets\21\Res\Lane01\deepLearnData\R001C001_mask.npy"
msk = np.load(path_mask).astype(int)
h = 2160
neighbor = []
for path in pathes:
    print(path)
    a = np.load(path) #a 是亮度矩阵。
    for i in range(msk.shape[0]):
        for j in range(msk.shape[1]):
            if msk[i,j] != 0:
                # here is a cluster:
                for ii in range(max(i-6,0),min(i+6,h-1)): #max min 防止溢出
                    for jj in range(max(j-6,0),min(j+6,h-1)):
                        if msk[ii,jj] !=0: # neighbor cluster
                            #find the cluster as the neighbor cluster cij j->i
                            #find the slop t for all these neigbor clusters to the center cluster.
                            #take its intensity
                            #calculate the distance and
                            neighbor.append()








