import numpy as np

img = np.load(r"E:\data\deepData\train\img\08.1h\009\150.npy")#4,h,w
pred = np.max(img)
lable = np.load(r"E:\data\deepData\train\label\08.1h\011\122.npy")
msk = np.load(r"E:\data\deepData\train\msk\08.1h\122.npy")
a,b = msk.shape
# for i in range(a):
#     for j in  range(b):
#         if msk[a,b] != 0:

