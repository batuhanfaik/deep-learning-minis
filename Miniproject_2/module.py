from typing import Optional, Any

from parameter import Parameter

class Module(object):
    def __init__(self, name: Optional[str] = None) -> None:
        self.name = name
        self.parameters = {}

    def register_parameter(self, name: str, parameter: Optional[Parameter] = None) -> None:
        self.parameters[name] = parameter

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return [(parameter.data, parameter.grad) for _, parameter in self.parameters.items() if parameter is not None]
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args)