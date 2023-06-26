import numpy as np
# import torch
import os
import glob


a =tuple('042222',)

print(int(a))
# #
# # a = np.loadtxt(r"E:\code\python_PK\callbase\datasets\highDens\result_outcorrd\Lane01\sfile\nearReadSet\R001C001_conv.npy")
# # b = np.loadtxt(r"E:\code\python_PK\callbase\datasets\highDens\result_outcorrd\Lane01\sfile\nearReadSet\R001C001.npy")
# c = np.loadtxt(r"E:\code\python_PK\callbase\datasets\highDens\result_outcorrd\Lane01\sfile\nearReadSet\R001C001_2_beforeLabel.txt")
# mean = np.load(r"E:\code\python_PK\bleeding\gtvalue\gt_from_img\R001C001_outlier_img_mean.npy")
# subloc = np.loadtxt(r"E:\code\python_PK\callbase\datasets\highDens\result_outcorrd\Lane01\sfile\nearReadSet\R001C001Outlier_Id.txt")
# label_vector = np.load(r"E:\code\python_PK\bleeding\gtvalue\label_vector\010_A_label_vector.npy")
# label_vector2 = np.load(r"E:\code\python_PK\bleeding\gtvalue\label_vector2\10_A_label_vector.npy")
# print(a[101254])
print("hello")



a = np.array([1,2.0,3])
if 2 in a:
    print("true")

# with open(r"E:\code\python_PK\callbase\datasets\highDens\result_outcorrd\Lane01\sfile/R001C001_1_A_my.temp",'r') as f:
#     a = f.readlines()[0]
#     b = float(a.split()[0])
#     print(b)
#
# a = np.load(r"E:\code\python_PK\callbase\datasets\highDens\result_outcorrd\Lane01\sfile/R001C001_1_A_my.temp")
# print(a[0])

# a = np.array([[1,5,3],[2,3,4]])
# c = np.argmax(a,axis=0)
# print(c)


# a = torch.tensor([[1,2,3],[2,3,4]])
# b = torch.tensor([[1,0,1],[1,1,0]])
# print(a*b)





# labelnpy = np.load("gtvalue/gtimg/Cyc001.npy")
# label= np.load("E:\code\python_PK\callbase\datasets\highDens\Result\Lane01\deepLearnData_final\Cyc001\label\R001C001_label.npy")
# for j in range(label.shape[0]):
#     for k in range(label.shape[1]):
#         if label[j, k] in [1, 2, 3, 4]:
#             print(label[j,k]," ",labelnpy[j,k,:])
















#
# a = np.array([2,1,2])
# b = np.array([3,3,3])
# test = np.array([[[1, 2, 4, 1],
#                   [2, 4, 2, 2],
#                   [2, 4, 2, 2]],
#
#                  [[2, 5, 4, 3],
#                   [3, 8, 2, 3],
#                   [2, 4, 2, 2]]])
# print(test.shape)
#
# print("axis =0:\n",np.sum(test,axis=0))
# print("axis =1:\n",np.sum(test,axis=1))
# print("axis =2:\n",np.sum(test,axis=2))

