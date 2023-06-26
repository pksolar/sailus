import torch
import torch.nn as nn
import torch.nn.functional as F

A = torch.ones(2,3)
B = 2*torch.ones(2,4)
C = torch.cat([A,B],dim=1)
print("c;:" ,C)
D = [C]
print("D:",D[0])
