from .. import Tensor


class Optimizer(object):
    def __init__(self, parameters, lr):
        if not isinstance(parameters, list):
            raise ValueError("`parameters` must be list")

        if len(parameters) == 0 or isinstance(parameters[0], Tensor):
            self.parameter_group = [{"param": parameters, "lr": lr}]
        elif isinstance(parameters[0], dict):
            self.parameter_group = parameters

    def step(self):
        raise NotImplementedError

