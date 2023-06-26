import numpy as np
error_12type = [[0,0,0,0],[0,0,1,0],[0,0,0,0],[0,0,0,0]]
with open("list_error.txt", 'w') as f:
    f.write(str(error_12type))
print(np.sum(np.array(error_12type)))