import numpy as np
import  glob
import os
dict_base = {'A':0,'C':1,'G':2,'T':3}
mean = np.load(r"E:\code\python_PK\bleeding\gtvalue\gt_from_img\R001C001_outlier_img_mean.npy")
paths = glob.glob(r"E:\code\python_PK\callbase\datasets\highDens\result_outcorrd\Lane01\sfile\R001C001*Coord.txt")
for path in paths:
    #path eg:E:\code\python_PK\callbase\datasets\highDens\result_outcorrd\Lane01\sfile\R001C001_40_A_otherCycChanCoord.txt
    print(path)
    baseId = np.loadtxt(path,usecols=(1))


    #img in which cycle and which channel:
    cycle = int(os.path.basename(path).split("_")[1])
    chn_name = os.path.basename(path).split("_")[2]
    chn = dict_base[chn_name] #0,1,2,3
    intensity_ = mean[cycle-1,chn,:]
    label_vector2 = np.where(baseId == 1,intensity_[0],baseId )
    label_vector2 = np.where(baseId == 2, intensity_[1],label_vector2 ) #
    label_vector2 = np.where(baseId == 3, intensity_[2],label_vector2 )
    label_vector2 = np.where(baseId == 4, intensity_[3],label_vector2 )
    np.save("gtvalue/label_vector/{}_{}_label_vector.npy".format(cycle,chn_name),label_vector2)
    print(cycle,chn)






    print("hello")
