import  numpy as np
import glob

paths = sorted(glob.glob(r"E:\code\python_PK\callbase\datasets\highDens\result_outcorrd\Lane01\sfile\nearReadSet\*Label.txt"))
for path in paths:
    print("path: ",path)
    a = np.loadtxt(path)
