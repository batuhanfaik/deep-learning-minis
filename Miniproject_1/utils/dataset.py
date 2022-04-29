import torch
import os


class NoiseDataset(torch.utils.data.Dataset):
    """
    Dataset for the project 1 containing noisy image pairs
    """
    def __init__(self, data_path: str = '../../miniproject_dataset/',
                 mode: str = 'train', device: str = 'cpu'):
        self.data_path = data_path
        self.mode = mode
        self.device = device
        self.data_input, self.data_target = self.__read_data('train')

    def __getitem__(self, index):
        return self.data_input[index], self.data_target[index]

    def __len__(self):
        return len(self.data)

    def __read_data(self, data_type: str = 'train'):
        if data_type == 'train':
            source, target = torch.load(os.path.join(self.data_path, 'train_data.pkl'),
                                        map_location=self.device)
        elif data_type == 'val':
            source, target = torch.load(os.path.join(self.data_path, 'val_data.pkl'),
                                        map_location=self.device)
        else:
            raise ValueError('data_type must be train or val')
        return source.float() / 255.0, target.float() / 255.0
