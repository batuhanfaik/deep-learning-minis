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

def conv2d(x: torch.Tensor, kernels: torch.Tesnor, padding: int):
    # input size is (batch_size x channels x height x width)
    N_in, C_in, H_in, W_in = x.shape
    
    # kernels are of size (out_channels, in_channels, kernel_height, kernel_width)
    N_ker, C_ker, H_ker, W_ker = kernels.shape
    
    # calculate output shape (based on https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
    H_out = H_in - H_ker + 1 + 2 * padding
    W_out = W_in - W_ker + 1 + 2 * padding
    
    # check whether the number of channels in the input is correct
    if C_in != C_ker: raise ValueError("Numbers of channels in the input and kernel are different.")
    
    # pad input
    if padding > 0:
        # can we modify the input tensor here or should we create a new one?
        input_ = torch.zeros(N_in, C_in, H_in, W_in)
        for n in range(N_in):
            for c in range(C_in):
                input_[n][c][padding:-padding][padding:-padding] = x[n][c]
    else:
        input_ = x
    
    # compute convolutions
    output = torch.zeros(N_in, N_ker, H_out, W_out)
    
    for n in range(N_in):
        for kernel in range(N_ker):
            for i in range(H_out):
                for j in range(W_out):
                    for c in range(C_in):
                        # this I am not sure of - probably use torch.tensordot?
                        output[n][kernel][i][j] += torch.mm(input_[n][c][???][???], kernels[kernel][c].T)
    
    
    return output