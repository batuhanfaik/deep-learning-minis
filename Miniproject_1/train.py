import itertools
from typing import Tuple
from pathlib import Path

import torch
import os

from utils.dataset import NoiseDataset
from utils.utils import psnr
from model import Model

DATA_PATH = 'miniproject_dataset/'
OUTPUT_MODEL_PATH = str(Path(__file__).parent / 'bestmodel.pth')
DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


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
    return source.float(), target.float()


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


def train(train_input, train_target, val_input, val_target, num_epochs=100,
          batch_size=64, validation_frequency=1, shuffle_data=True,
          use_augmentation=True, learning_rate=1e-2, wandb_name=None):
    if shuffle_data:
        train_rand_permutation = torch.randperm(train_input.shape[0])
        val_rand_permutation = torch.randperm(val_input.shape[0])
        train_input = train_input[train_rand_permutation]
        train_target = train_target[train_rand_permutation]
        val_input = val_input[val_rand_permutation]
        val_target = val_target[val_rand_permutation]

    if wandb_name is not None:
        import wandb
        wandb.init(project="dl_miniproject1", name=wandb_name, reinit=True,
                   config={"num_epochs": num_epochs,
                           "batch_size": batch_size,
                           "val_freq": validation_frequency,
                           "shuffle_data": shuffle_data,
                           "use_augmentation": use_augmentation,
                           "learning_rate": learning_rate},
                   )

    model = Model(lr=learning_rate, device=DEVICE)
    model.set_batch_size(batch_size)
    # OPTIONAL: Set the validation data and frequency
    model.set_val_data(val_input, val_target, validation_frequency=validation_frequency)
    # Train the model
    model.train(train_input, train_target, num_epochs,
                use_augmentation=use_augmentation, use_wandb=(wandb_name is not None))
    # Load the pretrained model
    # model.load_pretrained_model(OUTPUT_MODEL_PATH)
    # Evaluate the model
    prediction = model.predict(val_input)
    # Check the PSNR
    psnr_val = psnr(prediction / 255.0, val_target / 255.0, device=DEVICE)
    print(f'PSNR: {psnr_val:.6f} dB')

    if wandb_name is not None:
        wandb.log({"PSNR": psnr_val})
        # Save the best model
        model_path = str(
            Path(__file__).parent / f'{wandb_name}_bestmodel_{psnr_val:.4f}.pth')
        model.save_best_model(model_path)
        print(f'Saved model to `{model_path}`')
    return model, psnr_val


if __name__ == '__main__':
    # hyperparams = {
    #     'num_epochs': [50, 100, 150],
    #     'batch_size': [32, 64, 128],
    #     'shuffle_data': [True, False],
    #     'use_augmentation': [True, False],
    #     'learning_rate': [1e-2, 1e-3, 1e-4],
    # }
    hyperparams = {
        'num_epochs': [2, 5, 3],
        'batch_size': [64],
        'shuffle_data': [True],
        'use_augmentation': [True],
        'learning_rate': [1e-3],
    }
    train_input, train_target = get_data(mode='train', device=DEVICE)
    val_input, val_target = get_data(mode='val', device=DEVICE)

    # For all hyperparameter combinations
    test_name = 'bestrun'
    best_psnr = 25.4

    keys = hyperparams.keys()
    values = (hyperparams[key] for key in keys)
    combinations = [dict(zip(keys, combination)) for combination in
                    itertools.product(*values)]

    for idx, params in enumerate(combinations):
        print(f'Training with {params}')
        model, psnr_val = train(train_input, train_target, val_input, val_target,
                                wandb_name=f'{test_name}_Run-{idx + 1}',
                                **params)
        # Save best model
        if psnr_val > best_psnr:
            # Save the best model
            model.save_best_model(OUTPUT_MODEL_PATH)
            print(f'Saved model to `{OUTPUT_MODEL_PATH}`')
            best_psnr = psnr_val
        else:
            print(f'PSNR: {psnr_val:.6f} dB is not better than {best_psnr:.6f} dB')
