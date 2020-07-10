from ptsemseg.schedulers import get_scheduler
from torch.optim import SGD
import torch


x = torch.rand(100, requires_grad=True)
optm = SGD([x], 0.1)
s = get_scheduler(
    optm,
    {
        "name": "multi_step",
        "milestones": [80000, 90000],
        "warmup_iters": 10000,
        "warmup_mode": "constant",
        "warmup_factor": 0.2
    }
)

for i in range(300):
    print(s.get_lr())
    s.step()

