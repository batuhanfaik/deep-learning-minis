import torch

try:
    from .utils import zeros
except:
    from utils import zeros


def Parameter(data: torch.Tensor, requires_grad: bool = True) -> torch.Tensor:
    """Create a parameter tensor.
    This simply sets the requires_grad attribute
    and initializes the tensor grad to zeros.

    Args:
        data (torch.Tensor): Parameter data.
        requires_grad (bool, optional): Whether requires grad. Defaults to True.

    Returns:
        torch.Tensor: Parameter tensor.
    """
    data.requires_grad = requires_grad
    data.grad = zeros(data.shape)
    return data
