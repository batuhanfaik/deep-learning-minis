from typing import Union
from pathlib import Path

try:
    from .utils.gora import GORA
except:
    from utils.gora import GORA

import torch
import time


class Model:
    def __init__(self, learning_rate=1e-3) -> None:
        """
        Initialize model
            device: Device to use
            model: Model to use
            optimizer: Optimizer to use
            scheduler: Scheduler to use
            loss_fn: Loss function to use
            batch_size: Batch size to use
            validate_every: Validation frequency
            best_model: Dictionary to save best model
        """
        # Initialize model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = GORA().to(self.device)
        # Set the parameters
        self.optimizer = self.__get_optimizer(lr=learning_rate)
        self.scheduler = self.__get_scheduler(factor=0.5)
        self.loss_fn = self.__get_loss_fn().to(self.device)
        self.batch_size = 64
        # Validation data for performance tracking
        self.val_input, self.val_target = None, None
        self.validate_every = 0
        self.best_model = {'model': self.model.state_dict(), 'loss': float('inf')}

    def load_pretrained_model(self, ckpt_name: str = Path(__file__).parent / 'bestmodel.pth') -> None:
        """
        Load pretrained model
        ckpt_name: Path to checkpoint
        return: None
        """
        print(f'Loading pretrained model from {ckpt_name}')
        self.model.load_state_dict(torch.load(ckpt_name, map_location=self.device))

    def train(self, train_input: torch.Tensor, train_target: torch.Tensor,
              num_epochs: int = 1, use_augmentation: bool = False, use_wandb: bool = False) -> None:
        """
        Train model
        train_input: Input data
        train_target: Target data
        num_epochs: Number of epochs to train
        return: None
        """
        print('Training...')
        # Set model in training mode
        self.model.train()
        # If input is ByteTensor, convert to FloatTensor
        train_input = self.__check_input_type(train_input)
        train_target = self.__check_input_type(train_target)
        augmenter = None
        if use_augmentation:
            try:
                from .utils import AugmentedDataset
            except:
                from utils import AugmentedDataset
            augmenter = self.__get_augmenter(train_input, train_target)
        # Training loop
        loss_history = []
        running_loss = 0.0
        start_time = time.time()

        if use_wandb:
            import wandb

        for epoch in range(num_epochs):
            print(f'Epoch {epoch + 1} / {num_epochs}')
            # Minibatch loop
            for batch_idx in range(0, len(train_input), self.batch_size):
                # Get minibatch
                if use_augmentation:
                    batch_input, batch_target = augmenter[batch_idx:batch_idx + self.batch_size]
                else:
                    batch_input = train_input[batch_idx:batch_idx + self.batch_size]
                    batch_target = train_target[batch_idx:batch_idx + self.batch_size]
                batch_input, batch_target = batch_input.to(self.device), batch_target.to(self.device)
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

            if use_wandb:
                wandb.log({"train_loss": loss_history[-1]})

            # Validate if validation frequency is set, which requires a validation set
            if self.validate_every:
                if epoch % self.validate_every == self.validate_every - 1:
                    loss = self.validate(self.val_input, self.val_target)
                    print(f'\tValidation loss: {loss:.6f}')
                    if use_wandb:
                        wandb.log({"val_loss": loss})
                    self.scheduler.step(loss)
                    # Save model if validation loss is lower than the best model
                    if loss < self.best_model['loss']:
                        self.best_model['model'] = self.model.state_dict()

        end_time = time.time()
        print(f'Training time: {end_time - start_time:.2f}s')

    def validate(self, test_input: torch.Tensor, test_target: torch.Tensor) -> float:
        """
        Validate model
        test_input: Input data
        test_target: Target data
        return: Validation loss
        """
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
        """
        Predict using model
        test_input: Input data
        return: Predicted data
        """
        # Set model in evaluation mode
        self.model.eval()
        # If input is ByteTensor, convert to FloatTensor
        test_input = self.__check_input_type(test_input)
        # Predict on minibatches
        denoised_output = torch.empty(test_input.shape, dtype=torch.uint8).to(self.device)
        with torch.no_grad():
            for batch_idx in range(0, len(test_input), self.batch_size):
                # Get minibatch
                batch_input = test_input[batch_idx:batch_idx + self.batch_size].to(
                    self.device)
                # Forward pass
                output = self.model(batch_input)
                # Cutoff values to [0, 255] and convert to unsigned int tensor
                output = torch.clamp(output, 0, 255)
                # Save output
                denoised_output[batch_idx:batch_idx + self.batch_size] = output
        return denoised_output

    def __get_optimizer(self, lr: float = 1e-3) -> torch.optim:
        """
        Get optimizer
        lr: Learning rate
        return: Optimizer
        """
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
        """
        Get scheduler
        mode: Mode of scheduler
        factor: Factor of scheduler
        return: Scheduler
        """
        return torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode=mode,
                                                          factor=factor, patience=10,
                                                          verbose=False)

    def __get_loss_fn(self, loss_fn: Union[None, str] = None) \
            -> torch.nn.modules.loss:
        """
        Get loss function
        loss_fn: Loss function
        return: Loss function
        """
        if loss_fn is None or loss_fn == 'l2':
            return torch.nn.MSELoss().to(self.device)
        elif loss_fn == 'l1':
            return torch.nn.L1Loss().to(self.device)
        else:
            raise ValueError(f'Unknown loss function {loss_fn}')

    @staticmethod
    def __get_augmenter(source: torch.Tensor, target: torch.Tensor):
        """
        Get augmenter
        source: Source data
        target: Target data
        return: AugmentedDataset
        """
        augmenter = AugmentedDataset(source, target, autotransform=True)
        return augmenter

    @staticmethod
    def __check_input_type(tensor_input: torch.Tensor) -> torch.Tensor:
        """
        Check input type
        tensor_input: Input tensor
        return: Input tensor with correct type
        """
        # Convert Byte tensors to float if not already
        if isinstance(tensor_input, (torch.ByteTensor, torch.cuda.ByteTensor)):
            return tensor_input.float()
        return tensor_input

    def set_batch_size(self, batch_size: int) -> None:
        """
        Set batch size
        batch_size: Batch size
        return: None
        """
        self.batch_size = batch_size

    def set_val_data(self, val_input: torch.Tensor, val_target: torch.Tensor,
                     validation_frequency: int = 10) -> None:
        """
        Set validation data
        val_input: Input data
        val_target: Target data
        validation_frequency: Validation frequency
        return: None
        """
        self.val_input = val_input
        self.val_target = val_target
        self.validate_every = validation_frequency

    def save_best_model(self, path: str) -> None:
        """
        Save best model
        path: Path to save model
        return: None
        """
        torch.save(self.best_model['model'], path)
