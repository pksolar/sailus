import numpy as np
import matplotlib.pyplot as plt
arr = np.load("distance.npy")
t = np.nonzero(arr)
arr = arr[np.nonzero(arr)].ravel()
print(np.var(arr))
# 按照区间宽度 0.1 统计直方图
hist, bins = np.histogram(arr, bins=np.arange(0, np.max(arr) + 0.001, 0.001))

# 绘制直方图
plt.bar(bins[:-1], hist, width=0.1)
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.show()