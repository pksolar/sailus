import numpy as np

a = np.load(r"E:\data\highDensity\dense0.6\R001C001\res_deep\Lane01\deepLearnData/R001C001_mask.npy")
b = abs(a)
print(sum(sum(b)))