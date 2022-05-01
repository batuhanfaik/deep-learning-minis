import torch
from utils import get_gradient

ATTR_INPUTS = "inputs"
ATTR_OPERATION = "operation"

class GTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, x, metadata=None, *args, **kwargs):
        return super().__new__(cls, x, *args, **kwargs)
    
    def __init__(self, x, metadata=None):
        super().__init__()
        self.metadata = metadata or {ATTR_INPUTS: [], ATTR_OPERATION: None}

    def get_inputs(self):
        if self.metadata:
            return self.metadata.get(ATTR_INPUTS, [])
        return []
    
    def get_operation(self):
        if self.metadata:
            return self.metadata.get(ATTR_OPERATION)
        return None
    
    def backward(self, *gradients, **kwargs):
        gradient = get_gradient(gradients)
        operation = self.get_operation()

        if operation is not None:
            gradient = operation.backward(gradient)
 
        inputs = self.get_inputs()

        for input in inputs:
            if isinstance(input, GTensor):
                input.backward(gradient)
        

def make_gtensor(output, operation=None, inputs=None):
    if inputs is None:
        inputs = []
    
    if isinstance(inputs, torch.Tensor):
        inputs = [inputs]

    metadata = {ATTR_OPERATION: operation, ATTR_INPUTS: inputs}
    return GTensor(output, metadata)

        