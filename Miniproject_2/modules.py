from typing import Optional, Tuple, Union
from functools import reduce
import random
import math

import torch
from torch.nn.functional import fold, unfold

try:
    from .autograd import autograd_tensor, accumulate_grad
    from .module import Module
    from .parameter import Parameter
    from .functional import linear, relu, sigmoid, conv2d, conv_transpose2d, max_pool2d
    from .utils import check_inputs, get_gradient, zeros, ones, zeros_like, ones_like
except:
    from autograd import autograd_tensor, accumulate_grad
    from module import Module
    from parameter import Parameter
    from functional import linear, relu, sigmoid, conv2d, conv_transpose2d, max_pool2d
    from utils import check_inputs, get_gradient, zeros, ones, zeros_like, ones_like


class Sequential(Module):
    def __init__(self, *modules) -> None:
        super().__init__('Sequential')
        for module in modules:
            self.register_module(module)

    def forward(self, *input):
        check_inputs(input)
        output = input[0]

        for module in self.modules:
            output = module.forward(output)

        return autograd_tensor(output, self, input[0])

    def backward(self, *gradwrtoutput):
        output = get_gradient(gradwrtoutput)

        for module in self.modules[::-1]:
            output = module.backward(output)

        return output


class Linear(Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True) -> None:
        super().__init__('Linear')
        self.in_dim = in_dim
        self.out_dim = out_dim
        # TODO: implement Xavier initialization
        self.weight = Parameter(torch.empty((out_dim, in_dim)))
        self.bias = Parameter(torch.empty(out_dim)) if bias else None
        self.input_ = None
        self.register_parameter('weight', self.weight)
        self.register_parameter('bias', self.bias)

    def forward(self, *input):
        check_inputs(input)
        self.input_ = input[0]
        output = autograd_tensor(linear(self.input_, self.weight, self.bias),
                              self, self.input_)
        return output

    def backward(self, *gradwrtoutput):
        output_grad = get_gradient(gradwrtoutput)
        weight_grad = output_grad.T.mm(self.input_)
        bias_grad = output_grad.T.mm(ones((self.input_.shape[0], 1))).squeeze()
        input_grad = output_grad.mm(self.weight)
        accumulate_grad(self.weight, weight_grad)

        if self.bias is not None:
            accumulate_grad(self.bias, bias_grad)

        return input_grad


class Conv2d(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple],
                 stride: Union[int, Tuple] = 1, padding: Union[int, Tuple] = 0,
                 groups: int = 1, bias: bool = True, dilation: Union[int, Tuple] = 1, padding_mode: str = 'zeros') -> None:
        super().__init__("Conv2d")
        # Check if kernel size is a tuple of length 2 or int
        assert len(kernel_size) == 2 if isinstance(kernel_size, tuple) else \
            isinstance(kernel_size, int)
        # check if kernel size is int or tuple
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        # Check if stride is a tuple of length 2 or int
        assert len(stride) == 2 if isinstance(stride, tuple) else \
            isinstance(stride, int)
        # check if stride is int or tuple
        if isinstance(stride, int):
            stride = (stride, stride)
        # Check if padding is a tuple of length 2 or int
        assert len(padding) == 2 if isinstance(padding, tuple) else \
            isinstance(padding, int)
        # check if padding is int or tuple
        if isinstance(padding, int):
            padding = (padding, padding)
        # Check if dilation is a tuple of length 2 or int
        assert len(dilation) == 2 if isinstance(dilation, tuple) else \
            isinstance(dilation, int)
        # check if dilation is int or tuple
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        self.in_channels = in_channels
        self.out_channels = out_channels
        # if kernel size/padding is a single int = k, extend it to a (k x k) tuple
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.padding_mode = padding_mode
        self.groups = groups
        # initialize and register the kernels - we want out_channels kernels
        # each of size in_channels x kernel_h x kernel_w
        self.weight = Parameter(
            self.init_weights((self.out_channels, self.in_channels // self.groups,
                               self.kernel_size[0], self.kernel_size[1])))
        self.bias = Parameter(self.init_weights((self.out_channels,))) if bias else None
        self.register_parameter("weights", self.weight)
        self.register_parameter("bias", self.bias)

    def init_weights(self, shape):
        k = self.groups / (self.in_channels * self.kernel_size[0] * self.kernel_size[1])
        w = torch.tensor([random.uniform(-math.sqrt(k), math.sqrt(k)) for _ in range(reduce(lambda a,b: a*b, list(shape)))])
        return w.reshape(shape)

    def forward(self, *input_):
        check_inputs(input_)
        self.input_ = input_[0]
        output = autograd_tensor(conv2d(self.input_, self.weight, self.bias,
                                     padding=self.padding, stride=self.stride, dilation=self.dilation),
                              self, self.input_)
        return output

    def backward(self, *gradwrtoutput):
        grad = get_gradient(gradwrtoutput)
        batch_size, out_channels, out_height, out_width = grad.shape
        # compute the gradients wrt the output
        output_grad = grad.clone()
        # Take batch last
        for in_dim in range(len(output_grad.shape) - 1):
            output_grad = output_grad.transpose(in_dim, in_dim + 1)
        output_grad = output_grad.reshape(out_channels, -1)

        # Get the input
        tensor_in = self.input_
        tensor_in = unfold(tensor_in, self.kernel_size, self.dilation, self.padding,
                           self.stride)
        for in_dim in range(len(tensor_in.shape) - 1, 0, -1):
            tensor_in = tensor_in.transpose(in_dim, in_dim - 1)
        tensor_in = tensor_in.reshape(output_grad.shape[1], -1)

        # Weight and bias gradient
        weight_grad = output_grad.matmul(tensor_in).reshape(self.weight.shape)
        accumulate_grad(self.weight, weight_grad)

        if self.bias is not None:
            bias_grad = grad.sum(dim=(0, 2, 3))
            accumulate_grad(self.bias, bias_grad)

        # Input gradient
        in_height = (out_height - 1) * self.stride[0] - 2 * self.padding[0] + \
                     self.dilation[0] * (self.kernel_size[0] - 1) + 1
        in_width = (out_width - 1) * self.stride[1] - 2 * self.padding[1] + \
                    self.dilation[1] * (self.kernel_size[1] - 1) + 1
        input_grad = self.weight.reshape(out_channels, -1).T.matmul(output_grad)
        input_grad = input_grad.reshape(self.in_channels * self.kernel_size[0] *
                                        self.kernel_size[1], out_height * out_width,
                                        batch_size)
        for in_dim in range(len(input_grad.shape) - 1, 0, -1):
            input_grad = input_grad.transpose(in_dim, in_dim - 1)
        input_grad = fold(input_grad, (in_height, in_width), self.kernel_size,
                          self.dilation, self.padding, self.stride)
        input_grad = autograd_tensor(input_grad, self, self.input_)
        return input_grad


class TransposeConv2d(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple],
                 stride: Union[int, Tuple] = 1, padding: Union[int, Tuple] = 0,
                 groups: int = 1, bias: bool = True, dilation: Union[int, Tuple] = 1,
                 padding_mode: str = 'zeros') -> None:
        super().__init__('TransposeConv2d')
        # Check if kernel size is a tuple of length 2 or int
        assert len(kernel_size) == 2 if isinstance(kernel_size, tuple) else \
            isinstance(kernel_size, int)
        # check if kernel size is int or tuple
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        # Check if stride is a tuple of length 2 or int
        assert len(stride) == 2 if isinstance(stride, tuple) else \
            isinstance(stride, int)
        # check if stride is int or tuple
        if isinstance(stride, int):
            stride = (stride, stride)
        # Check if padding is a tuple of length 2 or int
        assert len(padding) == 2 if isinstance(padding, tuple) else \
            isinstance(padding, int)
        # check if padding is int or tuple
        if isinstance(padding, int):
            padding = (padding, padding)
        # Check if dilation is a tuple of length 2 or int
        assert len(dilation) == 2 if isinstance(dilation, tuple) else \
            isinstance(dilation, int)
        # check if dilation is int or tuple
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.weight = Parameter(self.init_weights((self.in_channels,
                                             self.out_channels // self.groups,
                                             self.kernel_size[0], self.kernel_size[1])))
        self.bias = Parameter(self.init_weights((self.out_channels,))) if bias else None
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.padding_mode = padding_mode
        self.input_ = None
        self.register_parameter("weight", self.weight)
        self.register_parameter("bias", self.bias)

    def init_weights(self, shape):
        k = self.groups / (self.out_channels * self.kernel_size[0] * self.kernel_size[1])
        w = torch.tensor([random.uniform(-math.sqrt(k), math.sqrt(k)) for _ in range(reduce(lambda a,b: a*b, list(shape)))])
        return w.reshape(shape)

    def forward(self, *input):
        check_inputs(input)
        check_inputs(input[0].shape, length=4)
        tensor_in = input[0]
        self.input_ = tensor_in
        out = conv_transpose2d(self.input_, weight=self.weight, bias=self.bias if self.bias is not None else None,
                              stride=self.stride, padding=self.padding, dilation=self.dilation)
        output = autograd_tensor(out, self, self.input_)
        return output

    def backward(self, *gradwrtoutput):
        grad = get_gradient(gradwrtoutput)
        batch_size, out_channels, out_height, out_width = grad.shape
        output_grad = unfold(grad, kernel_size=self.kernel_size, dilation=self.dilation,
                             padding=self.padding, stride=self.stride)
        weight_grad = output_grad.clone()

        for in_dim in range(len(weight_grad.shape) - 1, 0, -1):
            weight_grad = weight_grad.transpose(in_dim, in_dim - 1)

        tensor_in = self.input_

        for in_dim in range(len(tensor_in.shape) - 1):
            tensor_in = tensor_in.transpose(in_dim, in_dim + 1)

        tensor_in = tensor_in.reshape(self.in_channels, -1)
        weight_grad = tensor_in.matmul(
            weight_grad.reshape(tensor_in.shape[1], -1)).reshape(self.weight.shape)
        accumulate_grad(self.weight, weight_grad)

        if self.bias is not None:
            bias_grad = grad.sum(dim=(0, 2, 3))
            accumulate_grad(self.bias, bias_grad)

        in_height = (out_height + 2 * self.padding[0] - self.dilation[0] *
                     (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1

        in_width = (out_width + 2 * self.padding[1] - self.dilation[1] *
                    (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1

        input_grad = self.weight.reshape(self.in_channels, -1).matmul(output_grad)
        output = input_grad.reshape(batch_size, self.in_channels, in_height, in_width)
        output = autograd_tensor(output, self, self.input_)
        return output


class ReLU(Module):
    def __init__(self) -> None:
        super().__init__("ReLU")
        self.input_ = None

    def forward(self, *input):
        check_inputs(input)
        self.input_ = input[0]
        output = autograd_tensor(relu(self.input_), self, self.input_)
        return output

    def backward(self, *gradwrtoutput):
        grad = get_gradient(gradwrtoutput)
        input_grad = grad * (self.input_ > 0).int()
        return input_grad


class Sigmoid(Module):
    def __init__(self) -> None:
        super().__init__("Sigmoid")
        self.input_ = None

    def forward(self, *input):
        check_inputs(input)
        self.input_ = input[0]
        output = autograd_tensor(sigmoid(self.input_), self, self.input_)
        return output

    def backward(self, *gradwrtoutput):
        grad = get_gradient(gradwrtoutput)
        input_sigmoid = sigmoid(self.input_)
        input_grad = grad * input_sigmoid * (1 - input_sigmoid)
        return input_grad


class MSE(Module):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__("MSE")
        self.reduction = reduction
        self.input_ = None
        self.target = None

    def forward(self, *input):
        check_inputs(input, 2)
        self.input_ = input[0]
        self.target = input[1]

        if self.input_.shape != self.target.shape:
            raise ValueError("Input and target shapes should be same")

        error = (self.input_ - self.target) ** 2
        loss = error

        if self.reduction == "sum":
            loss = error.sum()
        elif self.reduction == "mean":
            loss = error.mean()

        return autograd_tensor(loss, self, [self.input_, self.target])

    def backward(self, *gradwrtoutput):
        grad = get_gradient(gradwrtoutput)
        input_grad = 2 * (self.input_ - self.target)

        if self.reduction == "mean":
            input_grad = input_grad / reduce(lambda a, b: a * b, self.input_.shape)

        return grad * input_grad


class MaxPool2d(Module):
    def __init__(self, kernel_size: Union[int, Tuple],
                 stride: Optional[Union[int, Tuple]] = None,
                 padding: Union[int, Tuple] = 0,
                 dilation: Union[int, Tuple] = 1) -> None:
        super().__init__("MaxPool2d")
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size,
                                                                    int) else kernel_size

        if stride is None:
            stride = kernel_size

        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else dilation

    def forward(self, *input):
        check_inputs(input)
        self.input_ = input[0]
        pool_output = max_pool2d(self.input_, kernel_size=self.kernel_size,
                                 stride=self.stride, padding=self.padding,
                                 dilation=self.dilation)
        output = autograd_tensor(pool_output, self, self.input_)
        return output

    def backward(self, *gradwrtoutput):
        output_grad = get_gradient(gradwrtoutput)
        input_grads = []
        N, C, H_in, W_in = self.input_.shape

        for ch in range(C):
            input_unfolded = unfold(self.input_[:, ch, :, :].unsqueeze(1),
                                  kernel_size=self.kernel_size,
                                  stride=self.stride, padding=self.padding,
                                  dilation=self.dilation)
            input_grad = zeros_like(input_unfolded)
            input_grad = input_grad.scatter(1,
                                            input_unfolded.argmax(dim=1, keepdims=True),
                                            1)
            input_grad = input_grad * output_grad[:, ch, :, :].reshape((N, 1, -1))
            input_grad = fold(input_grad, output_size=(H_in, W_in),
                              kernel_size=self.kernel_size,
                              stride=self.stride, padding=self.padding,
                              dilation=self.dilation)
            input_grads.append(input_grad)

        return torch.cat(input_grads, dim=1)


Upsampling = TransposeConv2d