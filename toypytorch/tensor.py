import numpy as np

from .autograd import engine
from .atuograd.function import Function
from .autograd.numpy import wrapper as anp
from .utils import topo_sort


class Tensor(object):
    """Box for np.ndarray.

    Anything you can do with an np.ndarray, you can do with an ArrayBox.
    """

    def __init__(self, value):
        if isinstance(value, list):
            value = np.asarray(value)
        while isinstance(value, self.__class__):
            value = value.get_value()

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
    T = property(lambda self: anp.transpose(self))
    def __len__(self): return len(self._value)
    def astype(self, *args, **kwargs): return anp._astype(self, *args, **kwargs)

    def __neg__(self): return anp.negative(self)
    def __add__(self, other): return anp.add(     self, other)
    def __sub__(self, other): return anp.subtract(self, other)
    def __mul__(self, other): return anp.multiply(self, other)
    def __pow__(self, other): return anp.power   (self, other)
    def __div__(self, other): return anp.divide(  self, other)
    def __mod__(self, other): return anp.mod(     self, other)
    def __truediv__(self, other): return anp.true_divide(self, other)
    def __matmul__(self, other): return anp.matmul(self, other)
    def __radd__(self, other): return anp.add(     other, self)
    def __rsub__(self, other): return anp.subtract(other, self)
    def __rmul__(self, other): return anp.multiply(other, self)
    def __rpow__(self, other): return anp.power(   other, self)
    def __rdiv__(self, other): return anp.divide(  other, self)
    def __rmod__(self, other): return anp.mod(     other, self)
    def __rtruediv__(self, other): return anp.true_divide(other, self)
    def __rmatmul__(self, other): return anp.matmul(other, self)
    def __eq__(self, other): return anp.equal(self, other)
    def __ne__(self, other): return anp.not_equal(self, other)
    def __gt__(self, other): return anp.greater(self, other)
    def __ge__(self, other): return anp.greater_equal(self, other)
    def __lt__(self, other): return anp.less(self, other)
    def __le__(self, other): return anp.less_equal(self, other)
    def __abs__(self): return anp.abs(self)
    def __hash__(self): return id(self)
    def __str__(self): return self.__repr__()
    def __repr__(self): 
        repr_text = self._value.__repr__().replace("array", "Tensor")
        func_name = ""
        if self.is_variable and self._autograd_meta and self._autograd_meta.get_function():
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
        return self._is_variable

    def set_to_variable(self):
        self._is_variable = True
        self._autograd_meta = AutogradMeta()
        self._autograd_meta.grad = anp.zeros_like(self._value)

    def disable_variable(self):
        self._is_variable = False

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
            grad = self.__class__(anp.ones((1, 1)))
        else:
            raise ValueError("Shape: {} of grad should be provided".format(self._value.shape))
            
        for grad_tensor in topo_sort(self):
            grad = grad if grad else grad_tensor.get_grad()
            while type(grad) == self.__class__:
                grad = grad.get_value()

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


# Set ArrayBox.<method name> = autograd.numpy.<function_name>
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
    setattr(Tensor, method_name, anp.__dict__[method_name])

# Flatten has no function, only a method.
setattr(Tensor, 'flatten', anp.__dict__['ravel'])


