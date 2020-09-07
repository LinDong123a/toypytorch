import numpy as np

from .autograd import functional
from .autograd.functional import *
from .tensor import Tensor


def zeros(*args, requires_grad: bool = False):
    return Tensor(np.zeros(args), requires_grad=requires_grad)

def ones(*args, requires_grad: bool = False):
    return Tensor(np.ones(args), requires_grad=requires_grad)

