import torch

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

def make_gtensor(output, operation=None, inputs=None):
    if inputs is None:
        inputs = []
    
    if isinstance(inputs, torch.Tensor):
        inputs = [inputs]

    metadata = {ATTR_OPERATION: operation, ATTR_INPUTS: inputs}
    return GTensor(output, metadata)

        