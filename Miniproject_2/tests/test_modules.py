import torch
import unittest

from modules import Linear, ReLU, Sigmoid, Sequential, MSELoss
from tensor import GTensor

class TestModules(unittest.TestCase):
    def test_linear(self):
        x = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        linear = Linear(2, 4)
        out = linear.forward(x)
        self.assertTrue(torch.equal(torch.mm(x, linear.weight.data.T) + linear.bias.data, out))
        self.assertTrue(isinstance(out, GTensor))
        self.assertTrue(torch.equal(out.get_inputs()[0], x))
        grads = torch.ones((2, 4))
        out = linear.backward(grads)
        self.assertTrue(torch.equal(torch.mm(grads, linear.weight.data), out))

    def test_relu(self):
        x = torch.tensor([2.0, -5.0, 3.0, 0.0])
        relu = ReLU()
        out = relu.forward(x)
        self.assertTrue(torch.equal(out, torch.tensor([2.0, 0.0, 3.0, 0.0])))
        self.assertTrue(isinstance(out, GTensor))
        self.assertTrue(torch.equal(out.get_inputs()[0], x))

        grads = torch.ones((4,))
        out = relu.backward(grads)
        self.assertTrue(torch.equal(out, torch.tensor([1.0, 0.0, 1.0, 0.0])))

    def test_sigmoid(self):
        x = torch.tensor([0, 1])
        sigmoid = Sigmoid()
        out = sigmoid.forward(x)
        self.assertTrue(torch.equal(out, torch.tensor([0.5, 1 / (1 + 1/torch.e)])))
        self.assertTrue(isinstance(out, GTensor))
        self.assertTrue(torch.equal(out.get_inputs()[0], x))

        grads = torch.ones((2,))
        out = sigmoid.backward(grads)
        self.assertTrue(torch.equal(out, torch.tensor([0.25, (1/torch.e) / (1 + 1/torch.e) ** 2])))
    
    def test_sequential(self):
        x = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        model = Sequential(Linear(2, 4), ReLU(), Linear(4, 1))
        out = model(x)
        self.assertEqual(out.shape, (2, 1))
        self.assertTrue(isinstance(out, GTensor))
        self.assertTrue(torch.equal(out.get_inputs()[0], x))

        grads = torch.ones((2, 1))
        out = model.backward(grads)
        self.assertEqual(out.shape, (2, 2))
    
    def test_mseloss(self):
        x = torch.tensor([2.0, -5.0, 3.0, 0.0])
        y = torch.tensor([2.0, 5.0, 4.0, 2.0])
        loss = MSELoss()
        out = loss(x, y)
        self.assertTrue(torch.equal(out, torch.tensor(105/4)))
        self.assertTrue(isinstance(out, GTensor))
        self.assertTrue(torch.equal(out.get_inputs()[0], x))
        self.assertTrue(torch.equal(out.get_inputs()[1], y))

        loss = MSELoss(reduction="sum")
        out = loss(x, y)
        self.assertTrue(torch.equal(out, torch.tensor(105.0)))

        loss = MSELoss(reduction=None)
        out = loss(x, y)
        self.assertTrue(torch.equal(out, torch.tensor([0.0, 100.0, 1.0, 4.0])))

        x = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        y = torch.tensor([[0], [1]])
        model = Sequential(Linear(2, 4), ReLU(), Linear(4, 1))
        out = model(x)
        criterion = MSELoss()
        loss = criterion(out, y)
        loss.backward()
