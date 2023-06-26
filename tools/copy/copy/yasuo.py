import numpy as np

data = np.load("img\R001C001_A.npy")

allzero = np.sum(data == 0, axis=1) # 加和值为128的所对应的行就是我们想要删除的
list_r = []
list_r2 = []
for i in range(len(allzero)):
    flag = allzero.tolist()[i]
    if flag==4096:        # 这里的数字需要被改为你的矩阵的实际的列数
        list_r.append(i) # 获得值为128的位置的索引
#print(list_r)
A_new = np.delete(data,np.s_[list_r],axis=0) # 通过索引删掉对应的行

allzero2 = np.sum(data == 0, axis=0)
for i in range(len(allzero2)):
    flag = allzero2.tolist()[i]
    if flag==2160:        # 这里的数字需要被改为你的矩阵的实际的列数
        list_r2.append(i) # 获得值为128的位置的索引
B_new = np.delete(A_new,np.s_[list_r2],axis=1) # 通过索引删掉对应的行
print(B_new.shape)



