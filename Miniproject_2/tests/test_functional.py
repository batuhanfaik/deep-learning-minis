import torch
import unittest

from functional import linear, relu, sigmoid


class TestFunctional(unittest.TestCase):
    def test_linear(self):
        x = torch.tensor([[1, 2, 3], [4, 5, 6]])
        w = torch.tensor([[1, 0, 1], [2, 2, 2]])
        out = linear(x, w)
        self.assertTrue(torch.equal(out, torch.tensor([[4, 12], [10, 30]])))

        out = linear(x, w, torch.tensor([1, 2]))
        self.assertTrue(torch.equal(out, torch.tensor([[5, 14], [11, 32]])))
    
    def test_relu(self):
        x = torch.tensor([2, -5, 3, 0])
        out = relu(x)
        self.assertTrue(torch.equal(out, torch.tensor([2, 0, 3, 0])))
    
    def test_sigmoid(self):
        x = torch.tensor([0, 1])
        out = sigmoid(x)
        self.assertTrue(torch.equal(out, torch.tensor([0.5, 1 / (1 + 1/torch.e)])))
