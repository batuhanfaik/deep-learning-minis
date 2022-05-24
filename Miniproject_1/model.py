from typing import Union
from pathlib import Path

try:
    from .utils import GORA
except:
    from utils import GORA

import torch
import time


class Model:
    def __init__(self) -> None:
        # Initialize model
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.model = GORA().to(self.device)
        # Set the parameters
        self.optimizer = self.__get_optimizer()
        self.scheduler = self.__get_scheduler(factor=0.5)
        self.loss_fn = self.__get_loss_fn().to(self.device)
        self.batch_size = 100
        # Validation data for performance tracking
        self.val_input, self.val_target = None, None
        self.validate_every = 0
        self.best_model = {'model': self.model.state_dict(), 'loss': float('inf')}

    def load_pretrained_model(self, ckpt_name: str = Path(__file__).parent / 'bestmodel.pth') -> None:
        print(f'Loading pretrained model from {ckpt_name}')
        self.model.load_state_dict(torch.load(ckpt_name, map_location=self.device))

    def train(self, train_input: torch.Tensor, train_target: torch.Tensor,
              num_epochs: int = 25) -> None:
        print('Training...')
        # Set model in training mode
        self.model.train()
        # If input is ByteTensor, convert to FloatTensor
        train_input = self.__check_input_type(train_input)
        train_target = self.__check_input_type(train_target)
        # Training loop
        loss_history = []
        running_loss = 0.0
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
                running_loss += loss.item()
                # Backward pass
                loss.backward()
                # Update parameters
                self.optimizer.step()
            # Append loss to history
            loss_history.append(running_loss / (len(train_input) / self.batch_size))
            running_loss = 0.0
            # Print loss
            print(f'\tLoss: {loss_history[-1]:.6f}')
            # Validate if validation frequency is set, which requires a validation set
            if self.validate_every:
                if epoch % self.validate_every == self.validate_every - 1:
                    loss = self.validate(self.val_input, self.val_target)
                    print(f'\tValidation loss: {loss:.6f}')
                    self.scheduler.step(loss)
                    # Save model if validation loss is lower than the best model
                    if loss < self.best_model['loss']:
                        self.best_model['model'] = self.model.state_dict()

        end_time = time.time()
        print(f'Training time: {end_time - start_time:.2f}s')

    def validate(self, test_input: torch.Tensor, test_target: torch.Tensor) -> float:
        print('Validating...')
        # Set model in evaluation mode
        self.model.eval()
        # If input is ByteTensor, convert to FloatTensor
        test_input = self.__check_input_type(test_input)
        test_target = self.__check_input_type(test_target)
        # Validation loop
        running_loss = 0.0
        with torch.no_grad():
            for batch_idx in range(0, len(test_input), self.batch_size):
                # Get minibatch
                batch_input = test_input[batch_idx:batch_idx + self.batch_size].to(
                    self.device)
                batch_target = test_target[batch_idx:batch_idx + self.batch_size].to(
                    self.device)
                # Forward pass
                output = self.model(batch_input)
                # Compute loss
                loss = self.loss_fn(output, batch_target)
                running_loss += loss.item()
            # Return loss
            return running_loss / (len(test_input) / self.batch_size)

    def predict(self, test_input: torch.Tensor) -> torch.Tensor:
        # Set model in evaluation mode
        self.model.eval()
        # If input is ByteTensor, convert to FloatTensor
        test_input = self.__check_input_type(test_input)
        # Predict on minibatches
        denoised_output = torch.empty(test_input.shape).to(self.device)
        with torch.no_grad():
            for batch_idx in range(0, len(test_input), self.batch_size):
                # Get minibatch
                batch_input = test_input[batch_idx:batch_idx + self.batch_size].to(
                    self.device)
                # Forward pass
                output = self.model(batch_input) * 255.0
                # Cutoff values to [0, 255]
                output = torch.clamp(output, 0, 255)
                # Convert to unsigned int tensor
                output = output.to(torch.uint8)
                # Save output
                denoised_output[batch_idx:batch_idx + self.batch_size] = output
        return denoised_output

    def __get_optimizer(self, lr: float = 1e-3) -> torch.optim:
        # Parameters are from paper 'Noise2Noise: Learning Image Restoration without
        # Clean Data' https://arxiv.org/abs/1803.04189
        return torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            betas=(0.9, 0.99),
            eps=1e-08,
        )

    def __get_scheduler(self, mode: str = 'min',
                        factor: float = 0.1) -> torch.optim.lr_scheduler:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode=mode,
                                                          factor=factor, patience=10,
                                                          verbose=False)

    def __get_loss_fn(self, loss_fn: Union[None, str] = None) \
            -> torch.nn.modules.loss:
        if loss_fn is None or loss_fn == 'l2':
            return torch.nn.MSELoss().to(self.device)
        elif loss_fn == 'l1':
            return torch.nn.L1Loss().to(self.device)
        else:
            raise ValueError(f'Unknown loss function {loss_fn}')

    @staticmethod
    def __check_input_type(tensor_input: torch.Tensor) -> torch.Tensor:
        # Convert Byte tensors to float if not already
        if isinstance(tensor_input, (torch.ByteTensor, torch.cuda.ByteTensor)):
            return tensor_input.float() / 255.0
        return tensor_input

    def set_batch_size(self, batch_size: int) -> None:
        self.batch_size = batch_size

    def set_val_data(self, val_input: torch.Tensor, val_target: torch.Tensor,
                     validation_frequency: int = 10) -> None:
        self.val_input = val_input
        self.val_target = val_target
        self.validate_every = validation_frequency

    def save_best_model(self, path: str) -> None:
        torch.save(self.best_model['model'], path)
