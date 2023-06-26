import numpy as np
import glob

E:\code\python_PK\callbase\datasets\highDens\result_outcorrd\Lane01\sfile\R002C034_41_G_my.temp
noMappedId = np.loadtxt("R001C001noMap.txt") # n,1  n : no mapped id.
paths = glob.glob(r"E:\code\python_PK\callbase\datasets\highDens\result_outcorrd\Lane01\sfile\*my.temp")
for path in paths:
    pos = np.loadtxt(path) # n,2  n ： the number of points