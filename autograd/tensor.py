from .function import Function
from .numpy import wrapper as anp
from .utils import topo_sort


class Tensor(object):
    """Box for np.ndarray.

    Anything you can do with an np.ndarray, you can do with an ArrayBox.
    """

    def __init__(self, value):
        self._value = value  # 节点保存的真实值
        self._autograd_meta: AutogradMeta = None # 保存与梯度相关的信息
        self._is_variable: bool = False  # 是否为可导的节点

    # ==============  重载，使得Tensor和np.narray有相似的接口 ================
    def __getitem__(A, idx): return A[idx]

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

    # ============== attributes access functions =================
    def set_autograd_meta(self, meta: AutogradMeta):
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

    def disable_variable(self):
        self._is_variable = False

    # ============= autograd related ====================
    def backward(self, grad):
        """ 反向传播的入口函数 """
        for grad_tensor in topo_sort(self):
            grad = grad if grad_tensor == self else grad_tensor.get_grad()

            grad_list = grad_tensor.get_function().backward(grad, self._value)
            for tensor, _grad in grad_list:
                tensor.add_grad(_grad)

    def add_grad(self, grad):
        self._autograd_meta.grad += grad


class AutogradMeta(object):
    """ 与自动求导相关的信息 """
    def __init__(self):
        self.function: Function = None  # 得到对应节点使用的操作
        self.grad = None               # 节点保存的梯度

    def get_function(self):
        return self.function

    def set_function(self, function: Function):
        self.function = function

