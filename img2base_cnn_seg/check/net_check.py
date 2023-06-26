import torch
import numpy as np
import torch.nn as nn

nn.CrossEntropyLoss()

target = np.array([
    [-1.0606, 1.5613, 1.2007, -0.2481],
    [-1.9652, -0.4367, -0.0645, -0.5104],
    [0.1011, -0.5904, 0.0243, 0.1002]
])
target = torch.tensor(target)
label = torch.tensor([0, 2, 1])

# 先取softmax，再取log操作

m = nn.LogSoftmax(dim=1)
# The negative log likelihood loss. It is useful to train a classification
# problem with `C` classes
loss = nn.NLLLoss()


tmp = m(target)
tmp_numpy = tmp.numpy()
output = loss(tmp, label)
print(output)
