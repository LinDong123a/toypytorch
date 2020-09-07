from . import Tensor


class Parameter(Tensor):
    def __init__(self, data, requires_grad: bool = True):
        super().__init__(data, requires_grad=requires_grad)

