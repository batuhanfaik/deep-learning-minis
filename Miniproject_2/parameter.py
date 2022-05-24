from functools import reduce
import torch

try:
    from .utils import zeros
except:
    from utils import zeros


def Parameter(data: torch.Tensor, requires_grad: bool = True):
    data.requires_grad = requires_grad
    data.grad = zeros(data.shape)
    return data

def accumulate_grad(tensor, *grads) -> torch.Tensor:
    if tensor.requires_grad:
        tensor.grad = tensor.grad + reduce(lambda a, b: a + b, grads)
    return tensor.grad

def zero_grad(tensor) -> None:
    tensor.grad.zero_()


