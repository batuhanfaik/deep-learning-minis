from typing import Optional
from functools import reduce

import torch

class Parameter:
    def __init__(self, data: Optional[torch.Tensor] = None, requires_grad: bool = True):
        if data is None:
            data = torch.empty(0)
        self.data = data
        self.requires_grad = requires_grad
        self.grad = torch.tensor(0)
    
    def accumulate_grad(self, *grads: torch.Tensor) -> torch.Tensor:
        if self.requires_grad:
            self.grad = self.grad + reduce(lambda a, b: a+b, grads)
        return self.grad
    
    def zero_grad(self) -> None:
        self.grad = torch.tensor(0)