import torch
from modules import Linear

x = torch.tensor([[1, 2], [3, 4]]).float()
linear = Linear(2, 4)
print(linear.weight.data)
print(linear.forward(x))
print(linear.backward(torch.ones((2, 4))))