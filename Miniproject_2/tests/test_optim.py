import unittest
import torch

from optim import SGD
from modules import Linear, ReLU, Sequential, MSELoss

class TestOptim(unittest.TestCase):
    def test_sgd(self):
        x = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        y = torch.tensor([[0], [1]])
        model = Sequential(Linear(2, 4), ReLU(), Linear(4, 1))
        optimizer = SGD(model.parameters(), lr=0.001)
        optimizer.zero_grad()
        out = model(x)
        criterion = MSELoss()
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()