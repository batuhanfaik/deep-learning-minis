import torch

from tensor import make_gtensor
from module import Module
from parameter import Parameter
from functional import linear, relu, sigmoid, conv2d
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
        output = make_gtensor(linear(self.input, self.weight.data, self.bias.data), self, self.input)
        return output
    
    def backward(self, *gradwrtoutput):
        grad = get_gradient(gradwrtoutput)
        weight_grad = torch.mm(grad.T, self.input)
        bias_grad = grad
        input_grad = torch.mm(grad, self.weight.data)
        self.weight.accumulate_grad(weight_grad)

        if self.bias is not None:
            self.bias.accumulate_grad(bias_grad)
        
        return input_grad


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
        input_grad = grad * torch.where(self.input > 0, 1, 0)
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
        input_grad = grad * input_sigmoid * (1-input_sigmoid)
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
            loss = torch.sum(error, dim=0)
        elif self.reduction == "mean":
            loss = torch.mean(error, dim=0)
        
        return make_gtensor(loss, self, [self.input, self.target])
    
    def backward(self, *gradwrtoutput):
        grad = get_gradient(gradwrtoutput)
        input_grad = 2 * (self.input - self.target)

        if self.reduction == "mean":
            input_grad = input_grad / len(self.input)
        
        return grad * input_grad
    

class Conv2D(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__("Conv2D")
        self.in_channels = in_channels
        self.out_channels = out_channels
        # if kernel size/padding is a single int = k, extend it to a (k x k) tuple
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.padding = padding
        
        # disregard stride and dilation for now...
        self.stride = stride
        self.dilation = dilation
        
        # initialize and register the kernels - we want out_channels kernels 
        # each of size in_channels x kernel_h x kernel_w
        self.kernels = Parameter(torch.rand((self.out_channels, self.in_channels) + self.kernel_size))
        self.register_parameter("kernels", self.kernels)
    
    def forward(self, *input_):
        check_inputs(input_)
        self.input_ = input_[0]                             
        output = make_gtensor(conv2d(self.input_, self.kernels, self.padding), self, self.input_)
        return output
    
    def backward(self, *gradwrtoutput):
        pass