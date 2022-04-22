from typing import Optional

import torch

def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Apply a linear transformation to the input x

    Args:
        x (torch.Tensor): Input tensor. Must be of shape (*, IN_DIM)
        weight (torch.Tensor): Weight matrix. Must be of shape (OUT_DIM, IN_DIM)
        bias (Optional[torch.Tensor], optional): Bias vector. Must be of shape (OUT_DIM). Defaults to None.

    Returns:
        torch.Tensor: Transformed input
    """
    output = torch.mm(x, weight.T)

    if bias is not None:
        output += bias
    
    return output

def relu(x: torch.Tensor) -> torch.Tensor:
    return torch.maximum(torch.tensor(0), x)

def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return x.sigmoid()
