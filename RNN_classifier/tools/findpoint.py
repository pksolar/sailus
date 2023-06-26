import numpy as np
a = np.load(r"E:\code\python_PK\callbase\datasets\08\Res\Lane01\deepLearnData\R001C001_mask.npy").astype(int)
for i in range(a.shape[0]):
    for j in range(a.shape[1]):
        if a[i,j] == 1:
            print(i,",",j)

