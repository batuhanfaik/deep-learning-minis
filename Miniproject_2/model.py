import torch
import time

from modules import ReLU, Sigmoid, ConvTranspose2d, Sequential, MSELoss, Conv2d
from optim import SGD

class Model:
    def __init__(self, learning_rate: float = 1e-3) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Sequential(
            Conv2d(3, 24, 3, stride=2),
            ReLU(),
            Conv2d(24, 24, 3, stride=2),
            ReLU(),
            ConvTranspose2d(24, 24, 3, stride=2),
            ReLU(),
            ConvTranspose2d(48, 48, 3, stride=2),
            Sigmoid())
        self.optimizer = SGD(self.model.parameters(), lr=learning_rate)
        self.criterion = MSELoss()
        self.validate_every = 0
        # self.best_model = {'model': self.model.state_dict(), 'loss': float('inf')}

    def load_pretrained_model(self, ckpt_name: str) -> None:
        print(f'Loading pretrained model from {ckpt_name}')
        self.model.load_state_dict(torch.load(ckpt_name, map_location=self.device))

    def train(self, train_input, train_target, num_epochs: int = 25) -> None:
        print('Training...')
        # Set model in training mode
        self.model.train()

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
                loss = self.criterion(output, batch_target)
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
                    # Save model if validation loss is lower than the best model
                    # if loss < self.best_model['loss']:
                    #     self.best_model['model'] = self.model.state_dict()

        end_time = time.time()
        print(f'Training time: {end_time - start_time:.2f}s')

    def validate(self, test_input: torch.Tensor, test_target: torch.Tensor) -> float:
        print('Validating...')
        # Set model in evaluation mode
        self.model.eval()
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
                loss = self.criterion(output, batch_target)
                running_loss += loss.item()
            # Return loss
            return running_loss / (len(test_input) / self.batch_size)

    def predict(self, test_input) -> torch.Tensor:
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

    def set_batch_size(self, batch_size: int) -> None:
        self.batch_size = batch_size

    def set_val_data(self, val_input: torch.Tensor, val_target: torch.Tensor,
                     validation_frequency: int = 10) -> None:
        self.val_input = val_input
        self.val_target = val_target
        self.validate_every = validation_frequency
