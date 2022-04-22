from typing import Tuple

import torch
import os

from utils import NoiseDataset
from utils import psnr
from model import Model

DATA_PATH = '../miniproject_dataset/'
OUTPUT_MODEL_PATH = 'bestmodel.pth'
SHUFFLE_DATA = True


def get_data(data_path: str = DATA_PATH, mode: str = 'train',
             device: torch.device = torch.device('cpu')) -> \
        Tuple[torch.Tensor, torch.Tensor]:
    """
    Loads the data from the given path.
    """
    if mode == 'train':
        return torch.load(os.path.join(data_path, 'train_data.pkl'),
                          map_location=device)
    elif mode == 'val':
        return torch.load(os.path.join(data_path, 'val_data.pkl'),
                          map_location=device)
    else:
        raise ValueError(f'Unknown data type {mode}')


def get_dataloaders(batch_size: int, shuffle: bool = True) -> \
        Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Returns the dataloaders for the training and validation data.
    """
    train_loader = torch.utils.data.DataLoader(NoiseDataset('train'),
                                               batch_size=batch_size,
                                               shuffle=shuffle)
    val_loader = torch.utils.data.DataLoader(NoiseDataset('val'),
                                             batch_size=batch_size,
                                             shuffle=shuffle)
    return train_loader, val_loader


if __name__ == '__main__':
    num_epochs = 2
    batch_size = 64
    # Validation step is optional
    validation_frequency = 10

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_input, train_target = get_data(mode='train', device=device)
    val_input, val_target = get_data(mode='val', device=device)

    if SHUFFLE_DATA:
        train_rand_permutation = torch.randperm(train_input.shape[0])
        val_rand_permutation = torch.randperm(val_input.shape[0])
        train_input = train_input[train_rand_permutation]
        train_target = train_target[train_rand_permutation]
        val_input = val_input[val_rand_permutation]
        val_target = val_target[val_rand_permutation]

    model = Model()
    model.set_batch_size(batch_size)
    # OPTIONAL: Set the validation data and frequency
    model.set_val_data(val_input, val_target, validation_frequency=validation_frequency)
    # Train the model
    model.train(train_input, train_target, num_epochs)
    # Evaluate the model
    prediction = model.predict(val_input)
    # Check the PSNR
    psnr_val = psnr(prediction, val_target)
    print(f'PSNR: {psnr_val:4f} dB')
    # Save the best model
    model.save_best_model(OUTPUT_MODEL_PATH)
    print(f'Saved model to `{OUTPUT_MODEL_PATH}`')
