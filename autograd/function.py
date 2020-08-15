from typing import List, Tuple, Dict
import numpy as anp

from .utils import upper_first_letter
from .tensor import Tensor


def get_func_class_name(func):
    return "{func}Function".format({"func": upper_first_letter(func.__name__)})


def unbroadcast(target, g, broadcast_idx=0):
    """Remove broadcasted dimensions by summing along them.

    When computing gradients of a broadcasted value, this is the right thing to
    do when computing the total derivative and accounting for cloning.
    """
    while anp.ndim(g) > anp.ndim(target):
        g = anp.sum(g, axis=broadcast_idx)
    for axis, size in enumerate(anp.shape(target)):
        if size == 1:
            g = anp.sum(g, axis=axis, keepdims=True)
    if anp.iscomplexobj(g) and not anp.iscomplex(target):
        g = anp.real(g)
    return g


def replace_zero(x, val):
    """Replace all zeros in 'x' with 'val'."""
    return anp.where(x, x, val)


class Function(object):
    """ 计算图中的结点，存储了计算图的入边，VJP的定义 """
    VJP_ALL = []

    def __init__(self):
        self._edge_list: List[Edge] = []
        self.tensor_dict: Dict[int, Tensor] = {}
        self.arg_vals = []

    def add_next_edge(self, next_edge):
        self._edge_list.append(next_edge)

    def add_next_edge_list(self, next_edge_list):
        for edge in next_edge_list:
            self.add_next_edge(edge)

    def set_tensor_args(self, tensor_args: List[Tensor]):
        for argnum, tensor in enumerate(tensor_args):
            self.tensor_dict[argnum] = tensor

    def set_args(self, args):
        self.arg_vals = args

    def get_all_next_fn(self):
        return [edge.function for edge in self._edge_list]

    def get_grad_in_input_nr(self, input_nr: int, grad, value):
        vjp = self.VJP_ALL[input_nr]
        return vjp(grad, value, *self.arg_vals)

    def get_all_next_fn_and_tensor(self):
        return [
            (edge.function, self.tensor_dict[edge.input_nr])
            for edge in self._edge_list
        ]

    def backward(self, grad, value):
        grad_list = []
        for edge in self._edge_list:
            grad_list.append((self.tensor_dict[edge.input_nr],
                              self.get_grad_in_input_nr(edge.input_nr, grad, value)))

        return grad_list


class Edge(object):
    """ 计算图中的边，由二元组(N, function)组成，表示作为目标结点的第N个输入 """
    def __init__(self, input_nr: int = None, func: Function = None):
        self.input_nr: int = input_nr
        self.function: Function = func

    def __eq__(self, obj):
        return self.function == obj.function and self.input_nr == obj.input_nr


class AddFunction(Function):
    VJP_ALL = [
        lambda g, ans, x, y: unbroadcast(x, g),
        lambda g, ans, x, y: unbroadcast(y, g),
    ]


class MultiplyFunction(Function):
    VJP_ALL = [
        lambda g, ans, x, y: unbroadcast(x, y * g),
        lambda g, ans, x, y: unbroadcast(y, x * g),
    ]


class SubtractFunction(Function):
    VJP_ALL = [
        lambda g, ans, x, y: unbroadcast(x, g),
        lambda g, ans, x, y: unbroadcast(y, -g),
    ]


class DivideFunction(Function):
    VJP_ALL = [
        lambda g, ans, x, y : unbroadcast(x,   g / y),
        lambda g, ans, x, y : unbroadcast(y, - g * x / y**2),
    ]


class True_divideFunction(Function):
    VJP_ALL = [
        lambda g, ans, x, y: unbroadcast(x, g / y),
        lambda g, ans, x, y: unbroadcast(y, - g * x / y ** 2)
    ]


class PowerFunction(Function):
    VJP_ALL = [
        lambda g, ans, x, y: unbroadcast(x, g * y * x ** anp.where(y, y - 1, 1.)),
        lambda g, ans, x, y:
            unbroadcast(y, g * anp.log(replace_zero(x, 1.)) * x ** y),
    ]


class NegativeFunction(Function):
    VJP_ALL = [
        lambda g, ans, x: -g,
    ]


class ExpFunction(Function):
    VJP_ALL = [
        lambda g, ans, x: ans * g
    ]


class LogFunction(Function):
    VJP_ALL = [
        lambda g, ans, x: g / x
    ]


class TanhFunction(Function):
    VJP_ALL = [
        lambda g, ans, x: g / anp.cosh(x) ** 2
    ]


class SinhFunction(Function):
    VJP_ALL = [
        lambda g, ans, x: g * anp.cosh(x)
    ]


class CoshFunction(Function):
    VJP_ALL = [
        lambda g, ans, x: g * anp.sinh(x)
    ]


class WhereFunction(Function):
    VJP_ALL = [
        lambda g, ans, c, x=None, y=None: anp.where(c, g, anp.zeros(g.shape)),
        lambda g, ans, c, x=None, y=None: anp.where(c, anp.zeros(g.shape), g),
    ]


class ReshapeFunction(Function):
    VJP_ALL = [
        lambda g, ans, x, shape, order=None:
            anp.reshape(g, anp.shape(x), order=order)
    ]

# ----- Dot grads -----


def _dot_vjp_0(g, ans, lhs, rhs):
    if max(anp.ndim(lhs), anp.ndim(rhs)) > 2:
        raise NotImplementedError("Current dot vjps only support ndim <= 2.")

    if anp.ndim(lhs) == 0:
        return anp.sum(rhs * g)
    if anp.ndim(lhs) == 1 and anp.ndim(rhs) == 1:
        return g * rhs
    if anp.ndim(lhs) == 2 and anp.ndim(rhs) == 1:
        return g[:, None] * rhs
    if anp.ndim(lhs) == 1 and anp.ndim(rhs) == 2:
        return anp.dot(rhs, g)
    return anp.dot(g, rhs.T)


def _dot_vjp_1(g, ans, lhs, rhs):
    if max(anp.ndim(lhs), anp.ndim(rhs)) > 2:
        raise NotImplementedError("Current dot vjps only support ndim <= 2.")

    if anp.ndim(rhs) == 0:
        return anp.sum(lhs * g)
    if anp.ndim(lhs) == 1 and anp.ndim(rhs) == 1:
        return g * lhs
    if anp.ndim(lhs) == 2 and anp.ndim(rhs) == 1:
        return anp.dot(g, lhs)
    if anp.ndim(lhs) == 1 and anp.ndim(rhs) == 2:
        return lhs[:, None] * g
    return anp.dot(lhs.T, g)


class DotFunction(Function):
    VJP_ALL = [
        _dot_vjp_0, _dot_vjp_1
    ]



