from typing import Optional, Any

from parameter import Parameter

class Module(object):
    def __init__(self, name: Optional[str] = None) -> None:
        self.name = name
        self._parameters = {}

    def register_parameter(self, name: str, parameter: Optional[Parameter] = None) -> None:
        self._parameters[name] = parameter

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError
    
    def parameters(self):
        return [parameter for _, parameter in self._parameters.items() if parameter is not None]

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args)