import collections
import functools
from typing import List, Tuple

from . import function
from .utils import subvals
from .tensor import Tensor, AutogradMeta


def primitive(f_raw):
    """ Wraps a function for recording computation graph info """
    @functools.wraps(f_raw)
    def f_wrapped(*args, **kwargs):
        tensor_args_list = find_tensor_args(args)
        if tensor_args_list:
            # 将数值从tensor中拿出来，使得可以正常用numpy进行处理
            argvals = subvals(
                args, 
                [(argnum, tensor.get_value()) for argnum, tensor in tensor_args_list]
            )

            ans = f_raw(*argvals, **kwargs)

            # 创建新的tensor来保存结果
            result_tensor = Tensor(ans)
            if is_func_output_variable(tensor_args_list):
                result_tensor.set_to_variable()
                
                # 保存计算图的相关信息
                grad_fn = getattr(function, function.get_func_class_name(f_raw))()
                grad_fn.add_next_edge_list(collect_next_edges_list(tensor_args_list))
                grad_fn.set_tensor_args([tensor for _, tensor in tensor_args_list])

                grad_meta = AutogradMeta()
                grad_meta.set_function(grad_fn)

                result_tensor.set_autograd_meta(grad_meta)

            return result_tensor
        else:
            return Tensor(f_raw(*args, **kwargs))
    return f_wrapped


def notrace_primitive(f_raw):
    """Wrap a raw numpy function by discarding boxes.

    Results are not boxed. Unboxing is a signal that the f_raw() is
    non-differentiable with respect to its arguments. Consider the computation,

    ```
    x = 1.5
    y = np.floor(x) + x
    ```

    What is the derivative of y wrt x? Autograd says 1. as np.floor has zero
    derivative near x=1.5.
    """
    @functools.wraps(f_raw)
    def f_wrapped(*args, **kwargs):
        tensor_args_list = find_tensor_args(args)
        if tensor_args_list:
            # 将数值从tensor中拿出来，使得可以正常用numpy进行处理
            argvals = subvals(
                args, 
                [(argnum, tensor.get_value()) for argnum, tensor in tensor_args_list]
            )

            ans = f_raw(*argvals, **kwargs)

            # 输出的节点将直接不可导
            return Tensor(ans)
        else:
            return Tensor(f_raw(*args, **kwargs))

    return f_wrapped


def find_tensor_args(args):
    return [(argnum, arg) for argnum, arg
            in enumerate(args) if isinstance(arg, Tensor) and arg.is_variable()]


def is_func_output_variable(args_list: List[Tuple[int, Tensor]]):
    # 输入的节点中，有任一一个为variable，则输出为variable
    return any([tensor.is_variable() for _, tensor in args_list])


def collect_next_edges_list(var_list: List[Tuple[int, Tensor]]):
    edge_list = []
    for argnum, var in var_list:
        if var.is_variable():
            edge_list.append(function.Edge(argnum, var.get_grad_fn()))

    return edge_list

