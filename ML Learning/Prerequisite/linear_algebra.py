import torch
X = torch.arange(24).reshape(2, 3, 4)
print(X)
print(X.mean(dtype=float))
print(X.sum(axis=2))
print(X.sum(axis=2, keepdims=True))