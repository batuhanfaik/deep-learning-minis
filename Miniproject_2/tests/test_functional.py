import torch
import unittest
import sys
sys.path.append("..")

from functional import linear, relu, sigmoid, max_pool2d


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

    def test_max_pool2d(self):
        x = torch.rand((2, 3, 4, 4))
        torch_out = torch.nn.functional.max_pool2d(x, kernel_size=2, stride=1)
        out = max_pool2d(x, kernel_size=2, stride=1)
        self.assertTrue(torch.isclose(torch_out, out).all())

        x = torch.rand((2, 3, 32, 32))
        torch_out = torch.nn.functional.max_pool2d(x, kernel_size=4, stride=2, padding=2, dilation=4)
        out = max_pool2d(x, kernel_size=4, stride=2, padding=2, dilation=4)
        self.assertTrue(torch.isclose(torch_out, out).all())

    def test_max_pool2d_padding(self):
        x = torch.rand((2, 3, 16, 16))
        torch_out = torch.nn.functional.max_pool2d(x, kernel_size=4, padding=2)
        out = max_pool2d(x, kernel_size=4, padding=2)
        self.assertTrue(torch.isclose(torch_out, out).all())

        x = torch.rand((2, 3, 16, 16))
        torch_out = torch.nn.functional.max_pool2d(x, kernel_size=4, padding=(1, 2))
        out = max_pool2d(x, kernel_size=4, padding=(1, 2))
        self.assertTrue(torch.isclose(torch_out, out).all())

    def test_max_pool2d_stride(self):
        x = torch.rand((2, 3, 4, 4))
        torch_out = torch.nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        out = max_pool2d(x, kernel_size=2, stride=2)
        self.assertTrue(torch.isclose(torch_out, out).all())

        x = torch.rand((2, 3, 4, 4))
        torch_out = torch.nn.functional.max_pool2d(x, kernel_size=2, stride=(2, 4))
        out = max_pool2d(x, kernel_size=2, stride=(2, 4))
        self.assertTrue(torch.isclose(torch_out, out).all())

        x = torch.rand((2, 3, 4, 4))
        torch_out = torch.nn.functional.max_pool2d(x, kernel_size=2, stride=None)
        out = max_pool2d(x, kernel_size=2, stride=None)
        self.assertTrue(torch.isclose(torch_out, out).all())

    def test_max_pool2d_dilation(self):
        x = torch.rand((2, 3, 4, 4))
        torch_out = torch.nn.functional.max_pool2d(x, kernel_size=2, dilation=2)
        out = max_pool2d(x, kernel_size=2, dilation=2)
        self.assertTrue(torch.isclose(torch_out, out).all())

        x = torch.rand((2, 3, 16, 16))
        torch_out = torch.nn.functional.max_pool2d(x, kernel_size=4, dilation=(1, 2))
        out = max_pool2d(x, kernel_size=4, dilation=(1, 2))
        self.assertTrue(torch.isclose(torch_out, out).all())