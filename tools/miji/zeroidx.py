import numpy  as np
import glob


totalpath_A  = glob.glob(r"E:\code\python_PK\tools\miji\flatten\pure\*.npy")
for i in totalpath_A:
    print(i)
    a = np.load(i)
    idx = np.where(a == 0)
    print(idx)
