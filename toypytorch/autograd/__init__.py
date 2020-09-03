import numpy as np

from .numpy.wrapper import *
from .tensor import Tensor


def zeros(shape, dtype=None):
    if dtype is not None:
        kwargs = {dtype: dtype}
    else:
        kwargs = {}

    return Tensor(np.zeros(shape, **kwargs))


def ones(shape, dtype=None):
    if dtype is not None:
        kwargs = {dtype: dtype}
    else:
        kwargs = {}

    return Tensor(np.ones(shape, **kwargs))


def rand(shape, dtype=None):
    if dtype is not None:
        kwargs = {dtype: dtype}
    else:
        kwargs = {}

    return Tensor(np.random.rand(shape, **kwargs))

