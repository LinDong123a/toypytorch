import numpy as np

from .base import Module
from ..parameter import Parameter
from .. import F


class Linear(Module):
    def __init__(self, in_features, out_features, bias: bool = True):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(np.ones((self.in_features, self.out_features)))
        if bias:
            self.bias = Parameter(np.zeros(self.out_features))
 
    def forward(self, input):
        """
        Args:
            input: [batch size, in_features]
        """
        out = F.matmul(input, self.weight)
        if hasattr(self, "bias"):
            out += self.bias

        return out

 

