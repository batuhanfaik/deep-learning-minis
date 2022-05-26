import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import random


class AugmentedDataset(Dataset):
    # Adapted from: https://discuss.pytorch.org/t/torchvision-transfors-how-to-perform-identical-transform-on-both-image-and-target/10606/5
    def __init__(self, source, target, autotransform=False, transform_per_image=False):
        self.source = source
        self.target = target
        self.autotransform = autotransform
        self.transform_per_image = transform_per_image
        # define the transformations (can be changed later on)
        self.transforms = [
            TF.vflip,
            TF.hflip,
            TF.adjust_brightness,
            TF.adjust_contrast,
            TF.adjust_gamma,
            TF.adjust_hue,
            TF.adjust_saturation,
            TF.rotate,
        ]
        # and the corresponding probabilities
        self.probas = [(0.5,),
                       (0.5,),
                       (0.5, 0.1),
                       (0.5, 0.1),
                       (0.5, 0.1),
                       (0.5, 0.05),
                       (0.5, 0.05),
                       (0.5, random.choice([90, 180, 270])),
                       ]

    def transform_pair(self, source_img, target_img):
        if self.transform_per_image:
            # batch size of source img and target img should be the same
            assert source_img.size() == target_img.size(), "source and target img should have the same size"
            source_img_ = torch.zeros(source_img.shape, dtype=source_img.dtype, device=source_img.device)
            target_img_ = torch.zeros(target_img.shape, dtype=target_img.dtype, device=target_img.device)
            # apply the transformations sequentially to every image in the batch
            for i in range(source_img.shape[0]):
                source_img_transformed = source_img[i]
                target_img_transformed = target_img[i]
                for transform, prob in zip(self.transforms, self.probas):
                    probability = prob[0]
                    args = prob[1:]
                    if random.random() < probability:
                        source_img_transformed = transform(source_img_transformed) if len(
                            args) == 0 else transform(source_img_transformed, *args)
                        target_img_transformed = transform(target_img_transformed) if len(
                            args) == 0 else transform(target_img_transformed, *args)
                source_img_[i] = source_img_transformed
                target_img_[i] = target_img_transformed
            return source_img_, target_img_
        # apply the transformations sequentially to the whole batch
        for transform, prob in zip(self.transforms, self.probas):
            probability = prob[0]
            args = prob[1:]
            if random.random() < probability:
                source_img = transform(source_img) if len(args) == 0 else transform(
                    source_img, *args)
                target_img = transform(target_img) if len(args) == 0 else transform(
                    target_img, *args)
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
