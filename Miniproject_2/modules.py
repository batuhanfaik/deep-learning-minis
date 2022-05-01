
import torch

from tensor import make_gtensor
from module import Module
from parameter import Parameter
from functional import linear, relu, sigmoid


def check_inputs(inputs, length=1):
    if len(inputs) != length:
        raise TypeError(f"Expected {length} inputs, got {len(inputs)}")


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
        check_inputs(gradwrtoutput)
        output = gradwrtoutput[0]

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
        check_inputs(gradwrtoutput)
        grad = gradwrtoutput[0]
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
        check_inputs(gradwrtoutput)
        grad = gradwrtoutput[0]
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
        check_inputs(gradwrtoutput)
        grad = gradwrtoutput[0]
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
            loss = torch.sum(error)
        elif self.reduction == "mean":
            loss = torch.mean(error)
        
        return make_gtensor(loss, self, [self.input, self.target])
    
    def backward(self, *gradwrtoutput):
        check_inputs(gradwrtoutput)
        grad = gradwrtoutput[0]
        input_grad = 2 * (self.input - self.target)

        if self.reduction == "sum":
            input_grad = torch.sum(input_grad)
        elif self.reduction == "mean":
            input_grad = torch.mean(input_grad)
        
        return grad * input_grad