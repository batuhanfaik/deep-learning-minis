from typing import List

from parameter import Parameter, zero_grad

class Optimizer(object):
    def __init__(self, parameters: List[Parameter]) -> None:
        self.parameters = parameters

    def zero_grad(self):
        raise NotImplementedError
    
    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, parameters: List[Parameter], lr: float = 0.01) -> None:
        super().__init__(parameters)
        self.lr = lr
    
    def zero_grad(self):
        for parameter in self.parameters:
            zero_grad(parameter)
        
    def step(self):
        for parameter in self.parameters:
            parameter.data = parameter.data - self.lr * parameter.grad