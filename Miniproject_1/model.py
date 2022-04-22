from typing import Optional, Tuple
from utils import GORA
from utils import NoiseDataset

import torch
import time
import os


class Model:
    def __init__(self) -> None:
        # Initialize model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = GORA().to(self.device)
        # Set the parameters
        self.optimizer = self.__get_optimizer()
        self.loss_fn = self.__get_loss_fn().to(self.device)
        self.batch_size = 32
        self.validate_every = 10
        # Get dataloaders
        self.train_loader, self.val_loader = self.__get_dataloaders()

    def load_pretrained_model(self, ckpt_name: str = 'bestmodel.pth') -> None:
        print(f'Loading pretrained model from {ckpt_name}')
        self.model.load_state_dict(torch.load(ckpt_name, map_location=self.device))

    def train(self, train_input: torch.Tensor, train_target: torch.Tensor, num_epochs: int) -> None:
        print('Training...')
        # Set model in training mode
        self.model.train()
        # Training loop
        start_time = time.time()
        for epoch in range(num_epochs):
            print(f'Epoch {epoch + 1} / {num_epochs}')
            # Minibatch loop
            for batch_idx in range(0, len(train_input), self.batch_size):
                # Get minibatch
                batch_input = train_input[batch_idx:batch_idx + self.batch_size].to(
                    self.device)
                batch_target = train_target[batch_idx:batch_idx + self.batch_size].to(
                    self.device)
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
            # Validate
            if epoch % self.validate_every == self.validate_every - 1:
                loss = self.validate(self.val_input, self.val_target)
                print(f'\tValidation loss: {loss:.4f}')

        end_time = time.time()
        print(f'Training time: {end_time - start_time:.2f}s')

    def validate(self, test_input: torch.Tensor, test_target: torch.Tensor) -> float:
        print('Validating...')
        # Set model in evaluation mode
        self.model.eval()
        # Predict on minibatches
        denoised_output = torch.empty(test_input.shape).to(self.device)
        with torch.no_grad():
            for batch_idx in range(0, len(test_input), self.batch_size):
                # Get minibatch
                batch_input = test_input[batch_idx:batch_idx + self.batch_size].to(
                    self.device)
                batch_target = test_target[batch_idx:batch_idx + self.batch_size].to(
                    self.device)
                # Forward pass
                output = self.model(batch_input)
                # Save output
                denoised_output[batch_idx:batch_idx + self.batch_size] = output
        # Compute loss
        loss = self.loss_fn(denoised_output, test_target)
        return loss.item()

    def predict(self, test_input: torch.Tensor) -> torch.Tensor:
        # Set model in evaluation mode
        self.model.eval()
        # Predict on minibatches
        denoised_output = torch.empty(test_input.shape).to(self.device)
        with torch.no_grad():
            for batch_idx in range(0, len(test_input), self.batch_size):
                # Get minibatch
                batch_input = test_input[batch_idx:batch_idx + self.batch_size].to(
                    self.device)
                # Forward pass
                output = self.model(batch_input)
                # Save output
                denoised_output[batch_idx:batch_idx + self.batch_size] = output
        return denoised_output

    def __get_optimizer(self, lr: float = 1e-3) -> torch.optim.Optimizer:
        # Parameters are from paper 'Noise2Noise: Learning Image Restoration without
        # Clean Data' https://arxiv.org/abs/1803.04189
        return torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            betas=(0.9, 0.99),
            eps=1e-08,
        )

    def __get_loss_fn(self, loss_fn: Optional[None, str] = None)\
            -> torch.nn.modules.loss:
        if loss_fn is None or loss_fn == 'l2':
            return torch.nn.MSELoss().to(self.device)
        elif loss_fn == 'l1':
            return torch.nn.L1Loss().to(self.device)
        else:
            raise ValueError(f'Unknown loss function {loss_fn}')

    def __get_dataloaders(self) -> Tuple[torch.utils.data.DataLoader,
                                         torch.utils.data.DataLoader]:
        train_loader = torch.utils.data.DataLoader(NoiseDataset('train'),
                                                   batch_size=self.batch_size,
                                                   shuffle=True)
        val_loader = torch.utils.data.DataLoader(NoiseDataset('val'),
                                                 batch_size=self.batch_size,
                                                 shuffle=True)
        return train_loader, val_loader
