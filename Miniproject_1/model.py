from typing import Optional
from utils import GORA

import torch


def __get_loss_fn(loss_fn: Optional[None, str] = None) -> torch.nn.modules.loss:
    if loss_fn is None or loss_fn == 'l2':
        return torch.nn.MSELoss()
    elif loss_fn == 'l1':
        return torch.nn.L1Loss()
    else:
        raise ValueError(f'Unknown loss function {loss_fn}')


class Model:
    def __init__(self) -> None:
        self.model = GORA()
        raise NotImplementedError

    def load_pretrained_model(self) -> None:
        raise NotImplementedError

    def train(self, train_input, train_target) -> None:
        raise NotImplementedError

    def predict(self, test_input) -> torch.Tensor:
        raise NotImplementedError

    def __get_optimizer(self, lr: float = 1e-3) -> torch.optim.Optimizer:
        # Parameters are from paper 'Noise2Noise: Learning Image Restoration without
        # Clean Data' https://arxiv.org/abs/1803.04189
        return torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            betas=(0.9, 0.99),
            eps=1e-08,
        )