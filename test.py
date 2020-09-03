import toypytorch as ttorch


a = ttorch.Tensor([[0.2, 0.1]])
b = ttorch.Tensor([[0.5], [0.5]])

a.set_to_variable()
b.set_to_variable()

c = ttorch.dot(a, b)
d = ttorch.exp(c)

print(d)

d.backward()

print(d.grad)
print(c.grad)
print(a.grad)
print(b.grad)
