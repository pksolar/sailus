import  numpy as np
import glob
pathes  = glob.glob(r"E:\code\python_PK\callbase\datasets\21\Res\Lane01\deepLearnData\*\intensity_norm\\R001C001_A.npy")
path_mask = r"E:\code\python_PK\callbase\datasets\21\Res\Lane01\deepLearnData\R001C001_mask.npy"
msk = np.load(path_mask).astype(int)
for path in pathes:
    print(path)
    a = np.load(path) #a 是亮度矩阵。
    for i in range(msk.shape[0]):
        for j in range(msk.shape[1]):
            if msk[i,j] != 0:
                print(a[i,j])


    print(a.shape)
