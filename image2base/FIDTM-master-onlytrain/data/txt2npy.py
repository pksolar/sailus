import numpy as np
import glob
paths = glob.glob(r"E:\code\python_PK\image2base\FIDTM-master\data\noname\poslist\*.txt")
for path in paths:
    poslist_total = []
    with open(path) as f:
        a  = f.readlines()
        for line in a:
            pos = line.split()
            poslist = [int(pos[1]),int(pos[0])]
            poslist_total.append(poslist)
        posnpy =  np.array(poslist_total)

    name = path.split("\\")[-1].split(".")[0].replace("img","array")
    savepath = r"E:\code\python_PK\image2base\FIDTM-master\data\noname\posnpy\\"+name+".npy"
    np.save(savepath,posnpy)




