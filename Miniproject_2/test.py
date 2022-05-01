
import torch

from modules import Linear, ReLU, Sequential, MSELoss

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
y = torch.tensor([[0], [1]])
model = Sequential(Linear(2, 4), ReLU(), Linear(4, 1))
out = model(x)
criterion = MSELoss()
loss = criterion(out, y)
loss.backward()