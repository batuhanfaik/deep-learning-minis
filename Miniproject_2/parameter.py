import torch

try:
    from .utils import zeros
except:
    from utils import zeros


def Parameter(data: torch.Tensor, requires_grad: bool = True):
    data.requires_grad = requires_grad
    data.grad = zeros(data.shape)
    return data
