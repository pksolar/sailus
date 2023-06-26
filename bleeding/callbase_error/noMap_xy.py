import numpy as np
nomap = np.load(r"E:\data\resize_test\08_resize_ori\res_deep\Lane01\deepLearnData\R001C001_mask.npy")
loc = np.where(nomap == -1)
with open("nomapxy.txt",'w') as f:
    for i in range(loc[0].shape[0]):
        f.write(str(int(loc[1][i])) + " " + str(int(loc[0][i])) + "\n")