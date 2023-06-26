import numpy as np
import  time
a = np.random.random((12,2160,4096))
s = time.time()
def norm(arr):# 对每个通道各自做归一化。
    for i in range(arr.shape[0]):
        channel = arr[i]
        # print("norm again")
        # min_val = np.percentile(channel,1)
        # max_val = np.percentile(channel,99)
        min_val = np.amin(channel)
        max_val = np.amax(channel)
        arr[i] = (channel - min_val) / (max_val - min_val)
    return arr
b = norm(a)
e = time.time()
k = time.time()
c = a/255
j = time.time()
print(j-k)
print("jhe",e-s)