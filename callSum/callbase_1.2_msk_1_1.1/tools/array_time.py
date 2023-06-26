import numpy as np
import time
a = np.random.rand(2160,4096)
lista = []
start = time.time()
for i in range(a.shape[0]):
    for j in range(a.shape[1]):
        lista.append(a[i,j])
end = time.time()
print(end-start)
