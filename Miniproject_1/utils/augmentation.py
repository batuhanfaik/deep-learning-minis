import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import random

class AugmentedDataset(Dataset):
    # Adapted from: https://discuss.pytorch.org/t/torchvision-transfors-how-to-perform-identical-transform-on-both-image-and-target/10606/5
    def __init__(self, source, target):
        self.source = source
        self.target = target
        
        # define the transformations (can be changed later on)
        self.transforms = [
            TF.vflip,
            TF.hflip
        ] 
        
        # and the corresponding probabilities
        self.probas = [0.5, 0.5]

    
    def transform_pair(self, source_img, target_img):
        # apply the transformations sequentially
        for transform, prob in zip(self.transforms, self.probas):
            if random.random() < prob:
                source_img = transform(source_img)
                target_img = transform(target_img)
                
        return source_img, target_img
    
    def __getitem__(self, i):
        x, y = self.source[i], self.target[i]
        return self.transform_pair(x, y)

    def __len__(self):
        return len(self.source)