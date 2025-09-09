import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

d2l.use_svg_display()

trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root="../data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=trans, download=True)

batch_size = 256

train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=4)

timer = d2l.Timer()
for X, y in train_iter:
    continue

print(timer.stop())