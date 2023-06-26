import numpy as np
import torch

inp = torch.arange(5*8).view(1, 1, 5, 8).float()
print('-------inp---------')
print(inp)
d = torch.linspace(-1, 1, 3)
meshw, meshh = torch.meshgrid((d, d))
print('---meshx---------meshy----')
print(meshw)
print(meshh)
print(((meshh+1)/2.)*(5.-1))# [-1,1]坐标转换为inp像素点坐标
print(((meshw+1)/2.)*(8.-1))
grid = torch.stack((meshw, meshh), 2).unsqueeze(0)
output = torch.nn.functional.grid_sample(inp, grid, mode='nearest', align_corners=True)
print('---output-----------')
print(output)