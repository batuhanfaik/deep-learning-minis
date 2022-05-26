import torch
from functools import partial

try:
    from .utils import get_gradient
except:
    from utils import get_gradient

ATTR_INPUTS = "inputs"
ATTR_OPERATION = "operation"
        
def get_inputs(tensor):
    if tensor.metadata:
        return tensor.metadata.get(ATTR_INPUTS, [])
    return []

def get_operation(tensor):
    if tensor.metadata:
        return tensor.metadata.get(ATTR_OPERATION)
    return None

def backward(tensor, *gradients, **kwargs):
    gradient = get_gradient(gradients)
    operation = get_operation(tensor)

    if operation is not None:
        gradient = operation.backward(gradient)

    inputs = get_inputs(tensor)

    for input_ in inputs:
        if input_.requires_grad:
            input_.backward(gradient)
    
    tensor.metadata = {ATTR_OPERATION: None, ATTR_INPUTS: []}
    tensor.backward = None
            
def autograd_tensor(tensor, operation=None, inputs=None):
    if inputs is None:
        inputs = []
    
    if isinstance(inputs, torch.Tensor):
        inputs = [inputs]
    
    if not tensor.requires_grad:
        tensor.requires_grad = True

    tensor.metadata = {ATTR_OPERATION: operation, ATTR_INPUTS: inputs}
    tensor.backward = partial(backward, tensor)

    return tensor

def accumulate_grad(tensor, grad) -> torch.Tensor:
    if tensor.requires_grad:
        tensor.grad = tensor.grad + grad
    return tensor.grad

def zero_grad(tensor) -> None:
    tensor.grad.zero_()