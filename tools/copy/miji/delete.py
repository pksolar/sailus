import numpy as np
a = np.random.randint(1,10,(5,5))
print(a)
d = np.array([0,1,0,1])
y = np.where(d!=0)

b = a[:,y].squeeze()
print(b.shape)
print(b)