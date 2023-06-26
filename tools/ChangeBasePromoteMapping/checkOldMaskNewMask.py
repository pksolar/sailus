import numpy as np
old = np.load(r"E:\data\resize_test\08_resize_ori\res_deep_intent\Lane01\deepLearnData\R001C001_mask.npy")
new = np.load(r"E:\data\autoupdate\msk\08hu_R001C001_msk.npy")
a = new - old
b = np.where(a==2)
c = np.where(a == -2)
d = np.where(a == 1)
print("hellow world")
