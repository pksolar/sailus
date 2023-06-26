import numpy as np
a = np.array([1,4])
b = np.array([4,1])
c = np.array([3,3])
g = np.sqrt((c -a ) ** 2 + (c - b ) ** 2)
print(g)
print(np.min(g))