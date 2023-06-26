import torch

a = torch.tensor([[1,4,5,2],
                  [1,4,5,2]])
b = torch.tensor([0,1,2,3,0,2,2,1,1,1])

print(a.shape)
_,t = torch.max(a,0)
print(t)
# a[a==0]=-10
# a[a==1]=0
# print(a)
# c = a+b
# d = c.flatten(0).numpy().tolist()t
# print(d)
# d_ =[]
# for i in d:
#     if i>=0:
#         d_.append(i)
# print(d_)