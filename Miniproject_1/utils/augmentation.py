import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import random


class AugmentedDataset(Dataset):
    # Adapted from: https://discuss.pytorch.org/t/torchvision-transfors-how-to-perform-identical-transform-on-both-image-and-target/10606/5
    def __init__(self, source, target, autotransform=False):
        self.source = source
        self.target = target
        self.autotransform = autotransform
        # define the transformations (can be changed later on)
        self.transforms = [
            TF.vflip,
            TF.hflip,
            TF.adjust_brightness,
            TF.adjust_contrast,
        ]
        # and the corresponding probabilities
        self.probas = [0.5, 0.5, (0.5, 0.25), (0.5, 0.25)]

    def transform_pair(self, source_img, target_img):
        # apply the transformations sequentially
        for transform, prob in zip(self.transforms, self.probas):
            if len(prob) > 1:
                probability = prob[0]
                args = prob[1:]
            else:
                probability = prob
                args = []
            if random.random() < probability:
                source_img = transform(source_img) if len(args) == 0 else transform(source_img, *args)
                target_img = transform(target_img) if len(args) == 0 else transform(target_img, *args)

        return source_img, target_img

    def __getitem__(self, i):
        x, y = self.source[i], self.target[i]
        if self.autotransform:
            return self.transform_pair(x, y)
        return x, y

    def __len__(self):
        return len(self.source)

    def __call__(self, *args, **kwargs):
        return self.source, self.target
