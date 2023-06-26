import torch
a = torch.randint(4,(3,3))
print(a)
print(a.shape)
a = a.long()

b = torch.nn.functional.one_hot(a,5)
print("bshape",b.shape)
b = b[:,:,:4]
b = b.permute(2,0,1).contiguous()
print(b)
print(b.shape)
