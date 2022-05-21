import imp
from numpy import packbits
import torch
from typing import Optional
from torch.nn.functional import fold, unfold
from math import floor

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

def conv2d(input_: torch.Tensor, kernels: torch.Tensor, padding: int, stride: int, dilation: int):
    # input size is (batch_size x channels x height x width)
    N_in, C_in, H_in, W_in = input_.shape

    # kernels are of size (out_channels, in_channels, kernel_height, kernel_width)
    C_out, C_ker, H_ker, W_ker = kernels.shape
    
    # calculate output shape (based on https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
    H_out = floor((H_in + 2 * padding - dilation  * (H_ker - 1) - 1) / stride + 1)
    W_out = floor((W_in + 2 * padding - dilation  * (W_ker - 1) - 1) / stride + 1)  

    # check whether the number of channels in the input is correct
    if C_in != C_ker: raise ValueError("Numbers of channels in the input and kernel are different.")
    packbits
    # compute convolutions
    input_unfolded = unfold(input_, kernel_size = (H_ker, W_ker), padding=padding, stride=stride, dilation=dilation)
    kernels_flattened = kernels.reshape(C_out, C_ker * H_ker * W_ker).T

    # print("Shapes: ", input_unfolded.shape, kernels_flattened.shape)
    output = torch.empty(N_in, C_out, H_out, W_out)

    for n in range(N_in):
        for k in range(C_out):
            row = input_unfolded[n].transpose(0, 1).mm(kernels_flattened[:, k].unsqueeze(1))
            output[n][k] = row.reshape(H_out, W_out)
    
    return output