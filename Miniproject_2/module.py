from typing import Optional, Any

PARAMETER_DELIMITER = "."
MODULE_DELIMITER = "#"

try:
    from .parameter import Parameter
    from .autograd import zero_grad
except:
    from parameter import Parameter
    from autograd import zero_grad


class Module(object):
    def __init__(self, name: Optional[str] = None) -> None:
        self.name = name
        self._parameters = {}
        self._training = True
        self._submodules = []

    def register_module(self, module):
        self._submodules.append(module)

    @property
    def modules(self):
        return self._submodules

    def register_parameter(self, name: str, parameter: Optional[Parameter] = None) -> None:
        self._parameters[name] = parameter

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError
    
    def parameters(self, recurse: bool = True):
        return [parameter for _, parameter in self.named_parameters(recurse=recurse)]

    def named_parameters(self, prefix: str = "", recurse: bool = True):
        if recurse and len(self.modules) > 0:
            named_parameters = []

            for i, module in enumerate(self.modules):
                named_parameters.extend(module.named_parameters(prefix=f"{prefix}{module.name}{MODULE_DELIMITER}{i}{PARAMETER_DELIMITER}",
                                                                recurse=recurse))
            
            return named_parameters

        return [(f"{prefix}{name}", parameter) for name, parameter in self._parameters.items() if parameter is not None]

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args)
    
    def train(self):
        self._training = True
    
    def eval(self):
        self._training = False
    
    def zero_grad(self):
        for parameter in self._parameters:
            zero_grad(parameter)
        
    def state_dict(self):
        named_parameters = self.named_parameters()
        return {name: parameter for name, parameter in named_parameters}
    
    def load_parameter(self, name, parameter):
        name_parts = name.split(PARAMETER_DELIMITER)

        if len(name_parts) == 1:
            self.register_parameter(name, parameter)
            return
        
        module_part = name_parts[0]
        module_name, module_num = module_part.split(MODULE_DELIMITER)
        module = self.modules[int(module_num)]

        if module.name != module_name:
            raise ValueError(f"Invalid state dict. {module_name} not found in submodules")

        module.load_parameter(name[name.find(PARAMETER_DELIMITER)+1:], parameter)

    def load_state_dict(self, state):
        for name, parameter in state.items():
            self.load_parameter(name, parameter)
        
        return self
    
    def to(self, device):
        parameters = self.parameters()

        for parameter in parameters:
            parameter.to(device)
        
        return self
