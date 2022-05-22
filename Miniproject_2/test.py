
import torch

from modules import MaxPool2d

from torch import autograd
x = torch.rand((2, 3, 4, 4), requires_grad=True)
torch_maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=1)
maxpool = MaxPool2d(kernel_size=2, stride=1)
torch_out = torch_maxpool(x)
out = maxpool(x)
# self.assertTrue(torch.isclose(torch_out, out).all())

torch.autograd.backward(torch_out, torch.ones_like(torch_out))
torch_grad = x.grad
grad = maxpool.backward(torch.ones_like(out))
print(torch_grad.shape, grad.shape)
print(torch_grad)
print(grad)
# self.assertTrue(torch.isclose(torch_grad, grad).all())