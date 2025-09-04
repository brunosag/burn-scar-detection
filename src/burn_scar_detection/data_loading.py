import os

import numpy as np
import rasterio
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
        t1 = TF.resized_crop(t1, i, j, h, w, size=(t1.shape[1], t1.shape[2]))
        t2 = TF.resized_crop(t2, i, j, h, w, size=(t2.shape[1], t2.shape[2]))
        mask = TF.resized_crop(mask, i, j, h, w, size=(mask.shape[1], mask.shape[2]))

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
            for i in range(4):  # Apply only to the 4 raw bands
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
    def __init__(self, t1_dir, t2_dir, mask_dir, augmentations=None):
        self.t1_dir = t1_dir
        self.t2_dir = t2_dir
        self.mask_dir = mask_dir
        self.augmentations = augmentations
        self.ids = sorted(
            [f.split('_')[-1].replace('.npy', '') for f in os.listdir(t1_dir)]
        )

    def __len__(self):
        return len(self.ids)

    def _normalize_patch(self, patch):
        normalized_patch = np.zeros_like(patch, dtype=np.float32)
        for i in range(patch.shape[0]):
            channel = patch[i, :, :]
            if np.any(~np.isfinite(channel)):
                channel = np.nan_to_num(channel, nan=0.0)
            p1, p99 = np.percentile(channel, [1, 99])
            clipped_channel = np.clip(channel, p1, p99)
            mean, std = clipped_channel.mean(), clipped_channel.std()
            normalized_patch[i, :, :] = (clipped_channel - mean) / (std + 1e-8)
        return normalized_patch

    def __getitem__(self, idx):
        id_ = self.ids[idx]

        t1_full = np.load(os.path.join(self.t1_dir, f'recorte_{id_}.npy'))
        t2_full = np.load(os.path.join(self.t2_dir, f'recorte_{id_}.npy'))

        with rasterio.open(os.path.join(self.mask_dir, f'recorte_{id_}.tif')) as src:
            mask = src.read(1)

        mask = np.where(mask > 0, 1.0, 0.0).astype(np.float32)

        t1_norm = self._normalize_patch(t1_full)
        t2_norm = self._normalize_patch(t2_full)

        t1 = torch.from_numpy(t1_norm).float()
        t2 = torch.from_numpy(t2_norm).float()
        y = torch.from_numpy(mask).float().unsqueeze(0)

        if self.augmentations:
            t1, t2, y = self.augmentations(t1, t2, y)

        return t1, t2, y
