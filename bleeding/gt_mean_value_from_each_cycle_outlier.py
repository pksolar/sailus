import os.path

import numpy as np
import glob

mean_array = np.zeros((100,4,4)) # 100 个cyc，4个channel的mean，每行16个值。
paths = sorted(glob.glob(r"E:\code\python_PK\callbase\datasets\highDens\result_outcorrd\Lane01\sfile\nearReadSet\*Label.txt"))
for path in paths:
    print("path: ",path)
    a = np.loadtxt(path)
    locA = np.where(a[4, :] == 0)
    locC = np.where(a[4, :] == 1)
    locG = np.where(a[4, :] == 2)
    locT = np.where(a[4, :] == 3)
    #call A时，A的值，统计以后求平均。
    intA_A_mean = np.mean(a[0, locA])
    # call A时，C的值，统计以后求平均。
    intA_C_mean = np.mean(a[1, locA])
    intA_G_mean = np.mean(a[2, locA])
    intA_T_mean = np.mean(a[3, locA])

    intC_A_mean = np.mean(a[0, locC])
    intC_C_mean = np.mean(a[1, locC])
    intC_G_mean = np.mean(a[2, locC])
    intC_T_mean = np.mean(a[3, locC])

    intG_A_mean = np.mean(a[0, locG])
    intG_C_mean = np.mean(a[1, locG])
    intG_G_mean = np.mean(a[2, locG])
    intG_T_mean = np.mean(a[3, locG])

    intT_A_mean = np.mean(a[0, locT])
    intT_C_mean = np.mean(a[1, locT])
    intT_G_mean = np.mean(a[2, locT])
    intT_T_mean = np.mean(a[3, locT])
    #change the name:
    num =int(os.path.basename(path).replace("R001C001_","").replace("_beforeLabel.txt",""))-1
    print("num:",num)
    mean_array[num,0,:] = np.array([intA_A_mean,intA_C_mean,intA_G_mean,intA_T_mean])
    mean_array[num,1,:] = np.array([intC_A_mean,intC_C_mean,intC_G_mean,intC_T_mean])
    mean_array[num, 2,:] = np.array([intG_A_mean,intG_C_mean,intG_G_mean,intG_T_mean])
    mean_array[num, 3,:] = np.array([intT_A_mean,intT_C_mean,intT_G_mean,intT_T_mean])

np.save("gtvalue/gt_from_img/R001C001_outlier_img_mean.npy",mean_array)



