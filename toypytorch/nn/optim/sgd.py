from .base import Optimizer


class SGD(Optimizer):
    def step(self):
        for param_info in self.parameter_group:
            for param in param_info["param"]:
                pass
