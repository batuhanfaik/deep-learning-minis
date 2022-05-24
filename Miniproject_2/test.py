import torch
from modules import Linear
from parameter import Parameter

x = torch.rand((3, 2), requires_grad=True)
torch_linear = torch.nn.Linear(2, 4)
linear = Linear(2, 4)
linear.weight = Parameter(torch_linear.weight)
linear.bias = Parameter(torch_linear.bias)
torch_out = torch_linear(x)
out = linear(x)
torch.autograd.backward(torch_out, torch.ones_like(torch_out))
grad = linear.backward(torch.ones_like(out))