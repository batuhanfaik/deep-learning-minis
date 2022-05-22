import torch
import unittest
import sys
sys.path.append("..")

from functional import linear, relu, sigmoid, max_pool2d


class TestFunctional(unittest.TestCase):
    def test_linear(self):
        x = torch.rand((3, 2))
        weight = torch.rand((4, 2))
        bias = torch.rand(4)
        torch_out = torch.nn.functional.linear(x, weight, bias)
        out = linear(x, weight, bias)
        self.assertTrue(torch.allclose(torch_out, out))
        self.assertEqual(torch_out.shape, out.shape)
    
    def test_relu(self):
        x = torch.rand((3, 2))
        torch_out = torch.nn.functional.relu(x)
        out = relu(x)
        self.assertTrue(torch.allclose(torch_out, out))
        self.assertEqual(torch_out.shape, out.shape)
    
    def test_sigmoid(self):
        x = torch.rand((3, 2))
        torch_out = torch.sigmoid(x)
        out = sigmoid(x)
        self.assertTrue(torch.allclose(torch_out, out))
        self.assertEqual(torch_out.shape, out.shape)

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