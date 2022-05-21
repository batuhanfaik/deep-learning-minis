import torch
import unittest
import sys
sys.path.append("..")

from parameter import Parameter


class TestParameter(unittest.TestCase):
    def test_basic(self):
        data = torch.tensor([1, 2, 3])
        parameter = Parameter(data)
        self.assertTrue(torch.equal(parameter.data, data))
        self.assertTrue(torch.equal(parameter.grad, torch.tensor(0)))
        self.assertEqual(parameter.requires_grad, True)
    
    def test_accumulate(self):
        data = torch.tensor([1, 2, 3])
        parameter = Parameter(data, requires_grad=False)
        self.assertTrue(torch.equal(parameter.grad, torch.tensor(0)))

        grads = [torch.tensor([2, 3, 5]), torch.tensor([1, 0, 1])]

        parameter.accumulate_grad(*grads)
        self.assertTrue(torch.equal(parameter.grad, torch.tensor(0)))

        parameter.requires_grad = True
        parameter.accumulate_grad(*grads)
        self.assertTrue(torch.equal(parameter.grad, torch.tensor([3, 3, 6])))
        
        parameter.zero_grad()
        self.assertTrue(torch.equal(parameter.grad, torch.tensor(0)))

