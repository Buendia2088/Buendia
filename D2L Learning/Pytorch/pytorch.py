import torch
from torch import nn

net = nn.Sequential(nn.Linear(8,64), nn.ReLU(), nn.Linear(64,10))
X = torch.rand(size=(2,8))
print(net[2].state_dict())