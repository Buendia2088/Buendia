import torch


X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
print(X)
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(Y)
Z = torch.cat((X, Y), dim = 1)
print(Z)
print(X == Y)
print(X.sum())

a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
print(a)
print(b)
print(a+b)

T = torch.zeros_like(Y)
print(id(T))
T[:] = T + Y
print(id(T))

