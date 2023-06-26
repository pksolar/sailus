import numpy as np
import shutil
import os
import glob

path_all =glob.glob(r"E:\code\python_PK\callbase\datasets\30\Res\Lane01\deepLearnData\*\label")
for path in path_all:
    name = path.split("\\")[-2]
    print(name)
    pathnew =  r"E:\code\python_PK\callbase\datasets\data\label\30\\"+name
    if os.path.exists(pathnew) is False:
        shutil.copytree(path,pathnew)