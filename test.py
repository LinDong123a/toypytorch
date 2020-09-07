import toypytorch as ttorch
import toypytorch.nn as nn
from toypytorch import functional as F

a = ttorch.Tensor([[0.2, 0.1]], requires_grad=True)
b = ttorch.Tensor([[0.5], [0.5]], requires_grad=True)

c = ttorch.dot(a, b)
print("c", c)
d = ttorch.exp(c)

print(d)

d.backward()

print(a.shape, b.shape, c.shape, d.shape)
print(d.grad)
print(c.grad)
print(a.grad)
print(b.grad)
