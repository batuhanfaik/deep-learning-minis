
import torch

from module import Module
from parameter import Parameter
from functional import linear, relu, sigmoid



class Sequential(Module):
    def __init__(self, *modules) -> None:
        super().__init__('Sequential')
        self.modules = modules
    
    def forward(self, *input):
        output = input

        for module in self.modules:
            if isinstance(output, torch.Tensor):
                output = module.forward(output)
            else:
                output = module.forward(*output)
        
        return output

    def backward(self, *gradwrtoutput):
        output = gradwrtoutput

        for module in self.modules[::-1]:
            if isinstance(output, torch.Tensor):
                output = module.backward(output)
            else:
                output = module.backward(*output)
        
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
        self.input = input
        outputs = [linear(x, self.weight.data, self.bias.data) for x in input]
        return outputs[0] if len(outputs) == 1 else outputs
    
    def backward(self, *gradwrtoutput):
        weight_grads = [torch.mm(grad.T, self.input[index]) for index, grad in enumerate(gradwrtoutput)]
        bias_grads = [grad for grad in gradwrtoutput]
        input_grads = [torch.mm(grad, self.weight.data) for grad in gradwrtoutput]
        self.weight.accumulate_grad(*weight_grads)

        if self.bias is not None:
            self.bias.accumulate_grad(*bias_grads)
        
        return input_grads[0] if len(input_grads) == 1 else input_grads


class ReLU(Module):
    def __init__(self) -> None:
        super().__init__("ReLU")
        self.input = None
    
    def forward(self, *input):
        self.input = input
        outputs = [relu(x) for x in input]
        return outputs[0] if len(outputs) == 1 else outputs
    
    def backward(self, *gradwrtoutput):
        input_grads = [grad * torch.where(self.input[index] > 0, 1, 0) for index, grad in enumerate(gradwrtoutput)]
        return input_grads[0] if len(input_grads) == 1 else input_grads


class Sigmoid(Module):
    def __init__(self) -> None:
        super().__init__("Sigmoid")
        self.input = None
    
    def forward(self, *input):
        self.input = input
        outputs = [sigmoid(x) for x in input]
        return outputs[0] if len(outputs) == 1 else outputs
    
    def backward(self, *gradwrtoutput):
        input_sigmoids = [sigmoid(x) for x in self.input]
        input_grads = [grad * input_sigmoids[index] * (1-input_sigmoids[index]) for index, grad in enumerate(gradwrtoutput)]
        return input_grads[0] if len(input_grads) == 1 else input_grads