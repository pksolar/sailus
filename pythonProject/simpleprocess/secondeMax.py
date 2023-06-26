import numpy as np
a = np.array([[10,4,6,7,4,56,7],
              [13,4,5,6,7,8,55]])
b = np.sort(a,0)
print(b)
c = np.sort(b,1)
print(c)
print(c[-1][-2])