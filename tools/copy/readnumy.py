import numpy as np
import glob
x = 3
y = 530
a = r"E:\code\python_PK\callbase\datasets\08\Res\Lane01\deepLearnData\Cyc001\intensity\R001C001_*.npy"
paths = glob.glob(a)
print("label:",np.load(r"E:\code\python_PK\callbase\datasets\08\Res\Lane01\deepLearnData\Cyc001\label\R001C001_label.npy")[x][y])
for path in paths:
   print(path)
   array_ = np.load(path)[x][y]

   print("point_value:",array_)
   print("------------------------------------------------------------------------------")
