import os

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset


class JointTransform:
    """Applies augmentations consistently to a pair of images and their mask."""

    def __init__(
        self, p_flip=0.5, p_photometric=0.5, scale=(0.8, 1.0), ratio=(0.75, 1.33)
    ):
        self.p_flip = p_flip
        self.p_photometric = p_photometric
        self.brightness = 0.1
        self.contrast = 0.1
        self.scale = scale
        self.ratio = ratio

    def __call__(self, t1, t2, mask):
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            t1, scale=self.scale, ratio=self.ratio
        )
        t1 = TF.resized_crop(t1, i, j, h, w, size=[t1.shape[1], t1.shape[2]])
        t2 = TF.resized_crop(t2, i, j, h, w, size=[t2.shape[1], t2.shape[2]])
        mask = TF.resized_crop(mask, i, j, h, w, size=[mask.shape[1], mask.shape[2]])

        if torch.rand(1) < self.p_flip:
            t1, t2, mask = TF.hflip(t1), TF.hflip(t2), TF.hflip(mask)
        if torch.rand(1) < self.p_flip:
            t1, t2, mask = TF.vflip(t1), TF.vflip(t2), TF.vflip(mask)

        k = np.random.randint(0, 4)
        if k > 0:
            angle = float(k * 90)
            t1, t2, mask = (
                TF.rotate(t1, angle),
                TF.rotate(t2, angle),
                TF.rotate(mask, angle),
            )

        if torch.rand(1) < self.p_photometric:
            brightness_factor = (
                torch.tensor(1.0)
                .uniform_(max(0, 1 - self.brightness), 1 + self.brightness)
                .item()
            )
            contrast_factor = (
                torch.tensor(1.0)
                .uniform_(max(0, 1 - self.contrast), 1 + self.contrast)
                .item()
            )
            # Apply only to the first 4 raw bands.
            for i in range(4):
                t1[i] = TF.adjust_contrast(
                    TF.adjust_brightness(t1[i].unsqueeze(0), brightness_factor),
                    contrast_factor,
                ).squeeze(0)
                t2[i] = TF.adjust_contrast(
                    TF.adjust_brightness(t2[i].unsqueeze(0), brightness_factor),
                    contrast_factor,
                ).squeeze(0)
        return t1, t2, mask


class BurnScarDataset(Dataset):
    def __init__(self, t1_feature_dir, t2_feature_dir, mask_dir, augmentations=None):
        self.t1_dir = t1_feature_dir
        self.t2_dir = t2_feature_dir
        self.mask_dir = mask_dir
        self.augmentations = augmentations
        self.ids = sorted(
            [
                f.replace('.npy', '')
                for f in os.listdir(t1_feature_dir)
                if f.endswith('.npy')
            ]
        )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_ = self.ids[idx]
        fname_npy = f'{id_}.npy'

        # 1. Load pre-processed feature stacks directly
        t1_full_norm = np.load(os.path.join(self.t1_dir, fname_npy))
        t2_full_norm = np.load(os.path.join(self.t2_dir, fname_npy))
        mask = np.load(os.path.join(self.mask_dir, fname_npy))

        # 2. Convert to tensors
        t1 = torch.from_numpy(t1_full_norm).float()
        t2 = torch.from_numpy(t2_full_norm).float()
        y = torch.from_numpy(mask).float().unsqueeze(0)

        # 3. Apply runtime augmentations
        if self.augmentations:
            t1, t2, y = self.augmentations(t1, t2, y)

        return t1, t2, y
