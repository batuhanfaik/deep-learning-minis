import torch
import unittest
import sys
sys.path.append("..")

from functional import linear, relu, sigmoid, conv2d



class TestFunctional(unittest.TestCase):
    def test_linear(self):
        x = torch.tensor([[1, 2, 3], [4, 5, 6]])
        w = torch.tensor([[1, 0, 1], [2, 2, 2]])
        out = linear(x, w)
        self.assertTrue(torch.isclose(out, torch.tensor([[4, 12], [10, 30]])).all())

        out = linear(x, w, torch.tensor([1, 2]))
        self.assertTrue(torch.isclose(out, torch.tensor([[5, 14], [11, 32]])).all())
    
    def test_relu(self):
        x = torch.tensor([2, -5, 3, 0])
        out = relu(x)
        self.assertTrue(torch.equal(out, torch.tensor([2, 0, 3, 0])))
    
    def test_sigmoid(self):
        x = torch.tensor([0, 1])
        out = sigmoid(x)
        self.assertTrue(torch.equal(out, torch.tensor([0.5, 1 / (1 + 1/torch.e)])))
    
    def test_conv2d_simple(self):
        x = torch.tensor([[[[ 0. , 1. , 2. , 3. ], [ 4., 5., 6., 7.], [ 8., 9., 10., 11.],
        [12.,13., 14., 15.]]]])
        kernels = torch.tensor([[[[0,1], [2,3]]]]).float()
        gt = torch.nn.functional.conv2d(x, kernels, stride=1, dilation=1, padding=0)
        out = conv2d(x, kernels, stride=1, dilation=1, padding=0)

    def test_conv2d_single_input(self):
        x = torch.rand((1, 1, 3, 3))
        kernels = torch.rand((1, 1, 2, 2))
        gt = torch.nn.functional.conv2d(x, kernels, stride=1, dilation=1, padding=0)
        out = conv2d(x, kernels, stride=1, dilation=1, padding=0)
        self.assertTrue(torch.isclose(out, gt).all())

    def test_conv2d_in_channels(self):
        x = torch.rand((1, 3, 5, 8))
        kernels = torch.rand((1, 3, 3, 3))
        gt = torch.nn.functional.conv2d(x, kernels, stride=1, dilation=1, padding=0)
        out = conv2d(x, kernels, stride=1, dilation=1, padding=0)
        self.assertTrue(torch.isclose(out, gt).all())

    def test_conv2d_out_channels(self):
        x = torch.rand((10, 3, 10, 10))
        kernels = torch.rand((4, 3, 2, 2))
        gt = torch.nn.functional.conv2d(x, kernels, stride=1, dilation=1, padding=0)
        out = conv2d(x, kernels, stride=1, dilation=1, padding=0)
        self.assertTrue(torch.isclose(out, gt).all())

    def test_test_conv2d_dilation(self):
        x = torch.rand((10, 3, 10, 10))
        kernels = torch.rand((4, 3, 2, 2))
        gt = torch.nn.functional.conv2d(x, kernels, stride=1, dilation=5, padding=0)
        out = conv2d(x, kernels, stride=1, dilation=5, padding=0)
        self.assertTrue(torch.isclose(out, gt).all())

    def test_conv2d_stride(self):
        x = torch.rand((10, 3, 10, 10))
        kernels = torch.rand((4, 3, 2, 2))
        gt = torch.nn.functional.conv2d(x, kernels, stride=3, dilation=1, padding=0)
        out = conv2d(x, kernels, stride=3, dilation=1, padding=0)
        self.assertTrue(torch.isclose(out, gt).all())
        
    def test_conv2d_padding(self):
        x = torch.rand((10, 3, 20, 10))
        kernels = torch.rand((4, 3, 2, 2))
        gt = torch.nn.functional.conv2d(x, kernels, stride=1, dilation=1, padding=4)
        out = conv2d(x, kernels, stride=1, dilation=1, padding=4)
        self.assertTrue(torch.isclose(out, gt).all())
        
    