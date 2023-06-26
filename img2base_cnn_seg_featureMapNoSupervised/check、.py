import numpy as np

a = np.array([[1,2,3],[0,1,1],[2,2,0]])
b = np.where(a!=0)
c = np.array(b).T
print("ddd")