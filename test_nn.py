import toypytorch as ttorch
from toypytorch import functional as F
import toypytorch.nn as nn


class Test(nn.Module):
    def __init__(self):
        self.linear1 = nn.Linear(10, 5)
        self.linear2 = nn.Linear(5, 1)

    def forward(self, input):
        return self.linear2(self.linear1(input))


test = Test()
out = test(ttorch.ones(10, 10))

scala = F.dot(ttorch.ones(1, 10), out)
print(scala)
scala.backward()

