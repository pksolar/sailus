import torch
a = torch.tensor([1,2]).unsqueeze(0)
b = torch.tensor([3,4]).unsqueeze(0)
c = torch.cat([a,b],dim=0)
print(c)