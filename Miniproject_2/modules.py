import torch
from torch.nn.functional import fold, unfold
from torch.nn.init import xavier_uniform_

from tensor import make_gtensor
from module import Module
from parameter import Parameter
from functional import linear, relu, sigmoid, convtranspose2d
from utils import check_inputs, get_gradient


class Sequential(Module):
    def __init__(self, *modules) -> None:
        super().__init__('Sequential')
        self.modules = modules

    def forward(self, *input):
        check_inputs(input)
        output = input[0]

        for module in self.modules:
            output = module.forward(output)

        return make_gtensor(output, self, input[0])

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
        self.input = None
        self.register_parameter('weight', self.weight)
        self.register_parameter('bias', self.bias)

    def forward(self, *input):
        check_inputs(input)
        self.input = input[0]
        output = make_gtensor(linear(self.input, self.weight.data, self.bias.data),
                              self, self.input)
        return output

    def backward(self, *gradwrtoutput):
        grad = get_gradient(gradwrtoutput)
        weight_grad = grad.T.mm(self.input)
        bias_grad = grad
        input_grad = grad.mm(self.weight.data)
        self.weight.accumulate_grad(weight_grad)

        if self.bias is not None:
            self.bias.accumulate_grad(bias_grad)

        return input_grad


class ConvTranspose2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 groups=1,
                 bias=True, dilation=1, padding_mode='zeros', device=None,
                 dtype=None) -> None:
        super().__init__('ConvTranspose2d')
        # Check if kernel size is a tuple of length 2 or int
        assert len(kernel_size) == 2 if isinstance(kernel_size, tuple) else isinstance(
            kernel_size, int)
        # check if kernel size is int or tuple
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        # Check if stride is a tuple of length 2 or int
        assert len(stride) == 2 if isinstance(stride, tuple) else isinstance(stride,
                                                                             int)
        # check if stride is int or tuple
        if isinstance(stride, int):
            stride = (stride, stride)
        # Check if padding is a tuple of length 2 or int
        assert len(padding) == 2 if isinstance(padding, tuple) else isinstance(padding,
                                                                               int)
        # check if padding is int or tuple
        if isinstance(padding, int):
            padding = (padding, padding)
        # Check if dilation is a tuple of length 2 or int
        assert len(dilation) == 2 if isinstance(dilation, tuple) else isinstance(
            dilation, int)
        # check if dilation is int or tuple
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.weight = torch.empty((self.in_channels, self.out_channels // self.groups,
                                   self.kernel_size[0], self.kernel_size[1]))
        self.bias = torch.empty(self.out_channels) if bias else None
        self.dilation = dilation
        self.padding_mode = padding_mode
        self.__init_params()

    def forward(self, *input):
        check_inputs(input[0].shape, length=4)
        tensor_in = input[0]
        output = make_gtensor(
            convtranspose2d(tensor_in, self.in_channels, self.out_channels,
                            self.kernel_size, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups))
        return output

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def __init_params(self):
        self.weight = Parameter(torch.rand(self.weight.shape))
        self.bias = Parameter(torch.rand(self.bias.shape)) if self.bias else None


class ReLU(Module):
    def __init__(self) -> None:
        super().__init__("ReLU")
        self.input = None

    def forward(self, *input):
        check_inputs(input)
        self.input = input[0]
        output = make_gtensor(relu(self.input), self, self.input)
        return output

    def backward(self, *gradwrtoutput):
        grad = get_gradient(gradwrtoutput)
        input_grad = grad * (self.input > 0).int()
        return input_grad


class Sigmoid(Module):
    def __init__(self) -> None:
        super().__init__("Sigmoid")
        self.input = None

    def forward(self, *input):
        check_inputs(input)
        self.input = input[0]
        output = make_gtensor(sigmoid(self.input), self, self.input)
        return output

    def backward(self, *gradwrtoutput):
        grad = get_gradient(gradwrtoutput)
        input_sigmoid = sigmoid(self.input)
        input_grad = grad * input_sigmoid * (1 - input_sigmoid)
        return input_grad


class MSELoss(Module):
    def __init__(self, reduction="mean") -> None:
        super().__init__("MSELoss")
        self.reduction = reduction
        self.input = None
        self.target = None

    def forward(self, *input):
        check_inputs(input, 2)
        self.input = input[0]
        self.target = input[1]
        error = (self.input - self.target) ** 2
        loss = error

        if self.reduction == "sum":
            loss = error.sum(dim=0)
        elif self.reduction == "mean":
            loss = error.mean(dim=0)

        return make_gtensor(loss, self, [self.input, self.target])

    def backward(self, *gradwrtoutput):
        grad = get_gradient(gradwrtoutput)
        input_grad = 2 * (self.input - self.target)

        if self.reduction == "mean":
            input_grad = input_grad / len(self.input)

        return grad * input_grad
