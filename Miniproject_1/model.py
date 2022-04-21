from typing import Optional
from utils import GORA

import torch
import time


class Model:
    def __init__(self) -> None:
        # Initialize model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = GORA().to(self.device)
        # Set the parameters
        self.optimizer = self.__get_optimizer()
        self.loss_fn = self.__get_loss_fn().to(self.device)
        self.n_epochs = 100
        self.batch_size = 32

    def load_pretrained_model(self) -> None:
        raise NotImplementedError

    def train(self, train_input: torch.Tensor, train_target: torch.Tensor) -> None:
        # Set model in training mode
        self.model.train()
        # Training loop
        start_time = time.time()
        for epoch in range(self.n_epochs):
            print(f'Epoch {epoch + 1} / {self.n_epochs}')
            # Minibatch loop
            for batch_idx in range(0, len(train_input), self.batch_size):
                # Get minibatch
                batch_input = train_input[batch_idx:batch_idx + self.batch_size].to(self.device)
                batch_target = train_target[batch_idx:batch_idx + self.batch_size].to(self.device)
                # Zero the gradients
                self.optimizer.zero_grad()
                # Forward pass
                output = self.model(batch_input)
                # Compute loss
                loss = self.loss_fn(output, batch_target)
                # Backward pass
                loss.backward()
                # Update parameters
                self.optimizer.step()
                # Print loss
                if batch_idx % 10 == 9:
                    print(
                        f'\tBatch {batch_idx + 1} / {len(train_input)}: {loss.item():.4f}')
        end_time = time.time()
        print(f'Training time: {end_time - start_time:.2f}s')

    def predict(self, test_input: torch.Tensor) -> torch.Tensor:
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

    @staticmethod
    def __get_loss_fn(loss_fn: Optional[None, str] = None) -> torch.nn.modules.loss:
        if loss_fn is None or loss_fn == 'l2':
            return torch.nn.MSELoss()
        elif loss_fn == 'l1':
            return torch.nn.L1Loss()
        else:
            raise ValueError(f'Unknown loss function {loss_fn}')
