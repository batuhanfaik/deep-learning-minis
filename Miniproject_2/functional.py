from typing import Optional

import torch
from torch.nn.functional import fold, unfold


def linear(x: torch.Tensor, weight: torch.Tensor,
           bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Apply a linear transformation to the input x

    Args:
        x (torch.Tensor): Input tensor. Must be of shape (*, IN_DIM)
        weight (torch.Tensor): Weight matrix. Must be of shape (OUT_DIM, IN_DIM)
        bias (Optional[torch.Tensor], optional): Bias vector. Must be of shape (OUT_DIM). Defaults to None.

    Returns:
        torch.Tensor: Transformed input
    """
    output = x.mm(weight.T)

    if bias is not None:
        output += bias

    return output


def convtranspose2d(tensor_in, in_channels, out_channels, kernel_size, weight, bias,
                    stride, padding, dilation, groups):
    batch_size, in_channels, in_height, in_width = tensor_in.shape
    # Take batch last
    for in_dim in range(len(tensor_in.shape) - 1):
        tensor_in = tensor_in.transpose(in_dim, in_dim + 1)
    tensor_in = tensor_in.reshape(in_channels, -1)
    output = weight.reshape(in_channels, -1).T.matmul(tensor_in)
    output_t = output.reshape(out_channels * kernel_size[0] * kernel_size[1],
                              in_height * in_width, batch_size)
    # Transpose to batch first
    for in_dim in range(len(output_t.shape) - 1, 0, -1):
        output_t = output_t.transpose(in_dim, in_dim - 1)
    out_height = (in_height - 1) * stride[0] - 2 * padding[0] + dilation[0] * (
            kernel_size[0] - 1) + 1
    out_width = (in_width - 1) * stride[1] - 2 * padding[1] + dilation[1] * (
            kernel_size[1] - 1) + 1
    output = fold(output_t, (out_height, out_width), kernel_size=kernel_size,
                  dilation=dilation, padding=padding, stride=stride)
    if bias:
        output += bias.reshape(1, -1)
    return output


def relu(x: torch.Tensor) -> torch.Tensor:
    return x.maximum(torch.tensor(0))


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return x.sigmoid()
