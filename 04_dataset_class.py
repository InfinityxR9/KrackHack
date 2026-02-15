# 4. Dataset Class

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# Import remap_mask from class ID mapping module
from class_id_mapping import remap_mask


class SegmentationDataset(Dataset):
    """
    Pairs images and masks by SORTED INDEX, not by filename.
    This is required because filenames differ between the two folders.
    """

    def __init__(self, image_dir: str, mask_dir: str, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # Sort both lists independently — alignment is by index
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))

        assert len(self.images) == len(self.masks), (
            f"Mismatch: {len(self.images)} images vs {len(self.masks)} masks"
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image as RGB (OpenCV loads BGR by default)
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask as uint16 grayscale — critical for values > 255 like 7100, 10000
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        # If mask was loaded as 3-channel, take first channel
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        # Remap to contiguous class IDs [0-9]
        mask = remap_mask(mask).astype(np.int64)

        # Apply augmentations — Albumentations treats mask as integer labels automatically
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]           # float32 tensor [3, H, W]
            mask = augmented["mask"]             # int64 tensor [H, W]

        # Ensure mask is LongTensor for CrossEntropyLoss / segmentation losses
        mask = mask.long() if isinstance(mask, torch.Tensor) else torch.from_numpy(mask).long()

        return image, mask
