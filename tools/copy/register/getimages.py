import cv2
"""

"""
import  numpy
import glob
import numpy as np

paths = glob.glob(r"E:\code\python_PK\callbase\datasets\30\Image\Lane01\*\R001C001_A.tif")
idx = 1
for path in paths:
    img =  cv2.imread(path,0)[500:1524,1500:2524]
    name = "{:04d}.jpg".format(idx)
    print(name)
    cv2.imwrite("images2/"+name,img)


    idx  += 1

