
import torch

from module import Module
from parameter import Parameter
from functional import linear


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
