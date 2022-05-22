import torch
from torch import autograd
import unittest
import sys

sys.path.append("..")

from modules import Linear, ConvTranspose2d, ReLU, Sigmoid, Sequential, MSELoss, MaxPool2d
from tensor import GTensor


class TestModules(unittest.TestCase):
    def test_linear(self):
        x = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        linear = Linear(2, 4)
        out = linear.forward(x)
        self.assertTrue(
            torch.equal(torch.mm(x, linear.weight.data.T) + linear.bias.data, out))
        self.assertTrue(isinstance(out, GTensor))
        self.assertTrue(torch.equal(out.get_inputs()[0], x))
        grads = torch.ones((2, 4))
        out = linear.backward(grads)
        self.assertTrue(torch.equal(torch.mm(grads, linear.weight.data), out))

    def test_convtranspose2d(self):
        # Create a 1x1x3x3 input tensor
        x = torch.tensor([[[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]]])
        # apply transpose convolution
        torch_conv = torch.nn.ConvTranspose2d(1, 1, kernel_size=2, stride=1, padding=0,
                                              bias=True)
        torch_conv.weight.data = torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]])
        torch_conv.bias.data = torch.tensor([5.0])
        # apply kernel
        torch_y = torch_conv(x)
        # apply transpose convolution with our implementation
        conv = ConvTranspose2d(1, 1, kernel_size=2, stride=1, padding=0, bias=True)
        conv.weight.data = torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]])
        conv.bias.data = torch.tensor([5.0])
        # apply kernel
        y = conv(x)
        self.assertTrue(torch.equal(y, torch_y))

    def test_convtranspose2d_backward(self):
        x = torch.tensor([[[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]]],
                         requires_grad=True)
        # apply transpose convolution using torch
        torch_conv = torch.nn.ConvTranspose2d(1, 1, kernel_size=2, stride=1, padding=0,
                                              bias=True)
        torch_conv.weight.data = torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]])
        torch_conv.bias.data = torch.tensor([5.0])
        torch_y = torch_conv(x)
        # autograd backwards
        autograd.backward(torch_y, torch.ones_like(torch_y))
        torch_x_grad = x.grad.clone()
        # zero gradients
        x.grad.zero_()
        # apply transpose convolution with our implementation
        conv = ConvTranspose2d(1, 1, kernel_size=2, stride=1, padding=0, bias=True)
        conv.weight.data = torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]])
        conv.bias.data = torch.tensor([5.0])
        y = conv(x)
        # our backwards
        our_x_grad = conv.backward(torch.ones_like(y))
        # compare gradients
        self.assertTrue(torch.equal(our_x_grad, torch_x_grad))

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
        self.assertTrue(torch.equal(out, torch.tensor([0.5, 1 / (1 + 1 / torch.e)])))
        self.assertTrue(isinstance(out, GTensor))
        self.assertTrue(torch.equal(out.get_inputs()[0], x))

        grads = torch.ones((2,))
        out = sigmoid.backward(grads)
        self.assertTrue(torch.equal(out, torch.tensor(
            [0.25, (1 / torch.e) / (1 + 1 / torch.e) ** 2])))

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
        self.assertTrue(torch.equal(out, torch.tensor(105 / 4)))
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

    def test_max_pool2d(self):
        x = torch.rand((2, 3, 4, 4))
        torch_maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=1)
        maxpool = MaxPool2d(kernel_size=2, stride=1)
        torch_out = torch_maxpool(x)
        out = maxpool(x)
        self.assertTrue(torch.isclose(torch_out, out).all())

        x = torch.rand((2, 3, 32, 32))
        torch_maxpool = torch.nn.MaxPool2d(kernel_size=4, stride=2, padding=2, dilation=4)
        maxpool = MaxPool2d(kernel_size=4, stride=2, padding=2, dilation=4)
        torch_out = torch_maxpool(x)
        out = maxpool(x)
        self.assertTrue(torch.isclose(torch_out, out).all())

    def test_maxpool2d_backward(self):
        x = torch.rand((2, 3, 32, 32), requires_grad=True)
        torch_maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=2)
        maxpool = MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=2)
        torch_out = torch_maxpool(x)
        out = maxpool(x)
        self.assertTrue(torch.isclose(torch_out, out).all())

        autograd.backward(torch_out, torch.ones_like(torch_out))
        torch_grad = x.grad
        grad = maxpool.backward(torch.ones_like(out))
        self.assertTrue(torch.isclose(torch_grad, grad).all())
