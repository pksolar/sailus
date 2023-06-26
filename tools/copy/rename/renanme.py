import os
import glob
import re
paths = glob.glob(r"E:\code\python_PK\bleeding\gtvalue\label_vector\*vector.npy")
for path in paths:
    basename = os.path.basename(path)
    num = basename.split("_")[0]
    path_new = path.replace(num,num.zfill(3))
    print(num)
    print(path_new)
    os.rename(path,path_new)