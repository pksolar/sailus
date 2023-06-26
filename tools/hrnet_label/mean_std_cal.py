import glob
"""
[0.02710745 0.03826159 0.03559004 0.03568636]
[0.02371992 0.02972373 0.03165162 0.02961854]

[6.91240028 9.75670515 9.07546121 9.10002067]
[6.04857953 7.5795504  8.07116209 7.55272754]

"""
import numpy as np
data = np.zeros((310,4,256,256))
a = glob.glob("img2base/imgdata/*.npy")
for i,j in enumerate(a):
    data[i] = np.load(j)
    if i == 309:
        break
channel_means = np.mean(data, axis=(0, 2, 3))
channel_stds = np.std(data, axis=(0, 2, 3))
print(channel_means/255)
print(channel_stds/255)