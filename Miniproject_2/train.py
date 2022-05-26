from typing import Tuple
from pathlib import Path

import torch
import os

from utils import psnr
from model import Model

# for testing
DATA_PATH = 'miniproject_dataset/'
OUTPUT_MODEL_PATH = str(Path(__file__).parent / 'bestmodel.pth')
SHUFFLE_DATA = True


def get_data(data_path: str = DATA_PATH, mode: str = 'train',
             device: torch.device = torch.device('cpu')) -> \
        Tuple[torch.Tensor, torch.Tensor]:
    """
    Loads the data from the given path.
    """
    if mode == 'train':
        source, target = torch.load(os.path.join(data_path, 'train_data.pkl'),
                                    map_location=device)
    elif mode == 'val':
        source, target = torch.load(os.path.join(data_path, 'val_data.pkl'),
                                    map_location=device)
    else:
        raise ValueError(f'Unknown data type {mode}')
    return source.float() / 255.0, target.float() / 255.0


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    num_epochs = 100
    batch_size = 1024
    # Validation step is optional
    validation_frequency = 10

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_input, train_target = get_data(mode='train', device=device)
    val_input, val_target = get_data(mode='val', device=device)

    if SHUFFLE_DATA:
        train_rand_permutation = torch.randperm(train_input.shape[0])
        val_rand_permutation = torch.randperm(val_input.shape[0])
        train_input = train_input[train_rand_permutation]
        train_target = train_target[train_rand_permutation]
        val_input = val_input[val_rand_permutation]
        val_target = val_target[val_rand_permutation]

    model = Model(learning_rate=1e-1)
    model.set_batch_size(batch_size)
    # OPTIONAL: Set the validation data and frequency
    model.set_val_data(val_input, val_target, validation_frequency=validation_frequency)
    # Load the pretrained model
    # model.load_pretrained_model(OUTPUT_MODEL_PATH)
    # Train the model
    model.train(train_input, train_target, num_epochs)
    # Evaluate the model
    prediction = model.predict(val_input) / 255.0
    # Check the PSNR
    psnr_val = psnr(prediction, val_target, device=device)
    print(f'PSNR: {psnr_val:.6f} dB')
    # Save the best model
    model.save_pretrained_model(OUTPUT_MODEL_PATH)
    print(f'Saved model to `{OUTPUT_MODEL_PATH}`')
