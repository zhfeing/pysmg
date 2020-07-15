from pysmg.schedulers import ConstantLR
from torch.optim import SGD
import torch

x = torch.rand(100, requires_grad=True)
optim = SGD([x], 1e-3)
s = ConstantLR(optim)

for i in range(10):
    if i == 5:
        for i in range(len(s.base_lrs)):
            s.base_lrs[i] /= 10
    print(s.get_last_lr())
    s.step()
