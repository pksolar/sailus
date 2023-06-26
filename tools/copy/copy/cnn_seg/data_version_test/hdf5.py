import numpy as np
import h5py
import time
s = time.time()
data = np.load('1.npy')
e = time.time()
print("npy",100*(e-s))
#
# with h5py.File('your_hdf5_file.h5', 'w') as hf:
#     hf.create_dataset('data', data=data, compression="gzip", compression_opts=9)
# print("hello world")
s_ = time.time()
with h5py.File('your_hdf5_file.h5', 'r') as hf:
    data2 = hf['data'][:]
e_ =time.time()

print("heelo",100*(e_-s_))
if data2.any() == data.any():
    print("same")
# 然后根据需要从data中读取子集，并进行预处理
