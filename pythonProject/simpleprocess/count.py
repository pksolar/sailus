import numpy as np

data=np.array([1,2,3,4,5,6])
# print("查看数组中各位置是否为1 \n",data == 1)

n = np.sum(data > 1)
print("数组中1的个数",n)