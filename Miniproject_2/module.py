from typing import Optional, Any

from parameter import Parameter, zero_grad

class Module(object):
    def __init__(self, name: Optional[str] = None) -> None:
        self.name = name
        self._parameters = {}
        self._training = True

    def register_parameter(self, name: str, parameter: Optional[Parameter] = None) -> None:
        self._parameters[name] = parameter

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError
    
    def parameters(self):
        return [parameter for _, parameter in self._parameters.items() if parameter is not None]

    def named_parameters(self):
        return [(name, parameter) for name, parameter in self._parameters.items() if parameter is not None]

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args)
    
    def train(self):
        self._training = True
    
    def eval(self):
        self._training = False
    
    def zero_grad(self):
        for parameter in self._parameters:
            zero_grad(parameter)