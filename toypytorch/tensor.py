import numpy as np

from .autograd import engine
from .autograd.function import Function
from .autograd import functional as F
from .utils import topo_sort


class Tensor(object):
    """Box for np.ndarray.

    Anything you can do with an np.ndarray, you can do with an Tensor.
    """

    def __init__(self, value, requires_grad: bool = False):
        if isinstance(value, list):
            value = np.asarray(value)
        while isinstance(value, self.__class__):
            value = value.get_value()

        self._value = value

        self.requires_grad = requires_grad
        if self.requires_grad:
            self.set_to_variable()
        else:
            self._value = value  # 节点保存的真实值
            self._autograd_meta: AutogradMeta = None  # 保存与梯度相关的信息
            self._is_variable: bool = False  # 是否为可导的节点

    # ==============  重载，使得Tensor和np.narray有相似的接口 ================
    def __getitem__(self, idx):
        # return self._value[idx]
        return engine.get_item(self, idx)

    # Constants w.r.t float data just pass though
    shape = property(lambda self: self._value.shape)
    ndim  = property(lambda self: self._value.ndim)
    size  = property(lambda self: self._value.size)
    dtype = property(lambda self: self._value.dtype)
    T = property(lambda self: F.transpose(self))
    def __len__(self): return len(self._value)
    def astype(self, *args, **kwargs): return F._astype(self, *args, **kwargs)

    def __neg__(self): return F.negative(self)
    def __add__(self, other): return F.add(     self, other)
    def __sub__(self, other): return F.subtract(self, other)
    def __mul__(self, other): return F.multiply(self, other)
    def __pow__(self, other): return F.power   (self, other)
    def __div__(self, other): return F.divide(  self, other)
    def __mod__(self, other): return F.mod(     self, other)
    def __truediv__(self, other): return F.true_divide(self, other)
    def __matmul__(self, other): return F.matmul(self, other)
    def __radd__(self, other): return F.add(     other, self)
    def __rsub__(self, other): return F.subtract(other, self)
    def __rmul__(self, other): return F.multiply(other, self)
    def __rpow__(self, other): return F.power(   other, self)
    def __rdiv__(self, other): return F.divide(  other, self)
    def __rmod__(self, other): return F.mod(     other, self)
    def __rtruediv__(self, other): return F.true_divide(other, self)
    def __rmatmul__(self, other): return F.matmul(other, self)
    def __eq__(self, other): return F.equal(self, other)
    def __ne__(self, other): return F.not_equal(self, other)
    def __gt__(self, other): return F.greater(self, other)
    def __ge__(self, other): return F.greater_equal(self, other)
    def __lt__(self, other): return F.less(self, other)
    def __le__(self, other): return F.less_equal(self, other)
    def __abs__(self): return F.abs(self)
    def __hash__(self): return id(self)
    def __str__(self): return self.__repr__()
    def __repr__(self): 
        repr_text = self._value.__repr__().replace("array", "Tensor")
        func_name = ""
        if self.requires_grad and self._autograd_meta and self._autograd_meta.get_function():
            func_name = ", grad_fn=<{}>".format(
                self._autograd_meta.get_function().__class__.__name__
            )

        return repr_text[:-1] + func_name + repr_text[-1:]

    # ============== attributes access functions =================
    def set_autograd_meta(self, meta):
        self._autograd_meta = meta

    def get_grad(self):
        return self._autograd_meta.grad

    def get_grad_fn(self):
        return self._autograd_meta.get_function()

    def get_value(self):
        return self._value

    def is_variable(self):
        return self.requires_grad

    def set_to_variable(self):
        self.requires_grad = True
        self._autograd_meta = AutogradMeta()
        self._autograd_meta.grad = np.zeros_like(self._value)

    @property
    def grad(self):
        try:
            return self._autograd_meta.grad.get_value()
        except:
            return self._autograd_meta.grad

    # ============= autograd related ====================
    def backward(self, grad=None):
        """ 反向传播的入口函数 """
        if grad is None and self._value.shape == (1, 1):
            grad = self.__class__(F.ones((1, 1)))
        else:
            raise ValueError("Shape: {} of grad should be provided".format(self._value.shape))

        if grad is not None and isinstance(grad, self.__class__):
            grad = grad.get_value()
            
        for grad_tensor in topo_sort(self):
            grad = grad if grad else grad_tensor.get_grad()

            grad_list = grad_tensor.get_grad_fn().backward(grad, self._value)
            for tensor, _grad in grad_list:
                tensor.add_grad(_grad)

            grad = None

    def add_grad(self, grad):
        self._autograd_meta.grad += grad


class AutogradMeta(object):
    """ 与自动求导相关的信息 """
    def __init__(self):
        self.function: Function = Function()  # 得到对应节点使用的操作
        self.grad = None               # 节点保存的梯度

    def get_function(self):
        return self.function

    def set_function(self, function: Function):
        self.function = function


engine.set_tensor_type(Tensor)
engine.set_grad_meta(AutogradMeta)


nondiff_methods = [
    'all',
    'any',
    'argmax',
    'argmin',
    'argpartition',
    'argsort',
    'nonzero',
    'searchsorted',
    'round']

diff_methods = [
    'clip',
    'compress',
    'cumprod',
    'cumsum',
    'diagonal',
    'max',
    'mean',
    'min',
    'prod',
    'ptp',
    'ravel',
    'repeat',
    'reshape',
    'squeeze',
    'std',
    'sum',
    'swapaxes',
    'take',
    'trace',
    'transpose',
    'var']

for method_name in nondiff_methods + diff_methods:
    setattr(Tensor, method_name, F.__dict__[method_name])

# Flatten has no function, only a method.
setattr(Tensor, 'flatten', F.__dict__['ravel'])


