import torch
import unittest
import sys
sys.path.append("..")

from parameter import Parameter, accumulate_grad, zero_grad


class TestParameter(unittest.TestCase):
    def test_basic(self):
        data = torch.tensor([1.0, 2.0, 3.0])
        parameter = Parameter(data)
        self.assertTrue(torch.equal(parameter.data, data))
        self.assertTrue(torch.equal(parameter.grad, torch.zeros_like(parameter.grad)))
        self.assertEqual(parameter.requires_grad, True)
    
    def test_accumulate(self):
        data = torch.tensor([1.0, 2.0, 3.0])
        parameter = Parameter(data, requires_grad=False)
        self.assertTrue(torch.equal(parameter.grad, torch.zeros_like(parameter.grad)))

        grads = [torch.tensor([2.0, 3.0, 5.0]), torch.tensor([1.0, 0.0, 1.0])]

        accumulate_grad(parameter, *grads)
        self.assertTrue(torch.equal(parameter.grad, torch.zeros_like(parameter.grad)))

        parameter.requires_grad = True
        accumulate_grad(parameter, *grads)
        self.assertTrue(torch.equal(parameter.grad, torch.tensor([3.0, 3.0, 6.0])))
        
        zero_grad(parameter)
        self.assertTrue(torch.equal(parameter.grad, torch.zeros_like(parameter.grad)))

