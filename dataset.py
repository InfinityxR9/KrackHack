"""
Dataset class for off-road desert semantic segmentation.

Key design decisions:
- Images and masks have DIFFERENT filenames but are aligned by sorted order.
- Masks are uint16 PNGs with non-contiguous class IDs that must be remapped to [0-9].
- Albumentations handles joint image+mask transforms so spatial augmentations stay consistent.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


# Original class IDs found in the masks -> contiguous labels [0-9]
CLASS_IDS = [100, 200, 300, 500, 550, 600, 700, 800, 7100, 10000]
NUM_CLASSES = len(CLASS_IDS)

# Build a lookup table for fast remapping (max value is 10000, so table size is 10001)
_REMAP_LUT = np.full(10001, 0, dtype=np.int64)  # default to 0 for any unexpected value
for contiguous_label, original_id in enumerate(CLASS_IDS):
    _REMAP_LUT[original_id] = contiguous_label


def remap_mask(mask: np.ndarray) -> np.ndarray:
    """Remap raw mask pixel values to contiguous class indices [0, NUM_CLASSES-1]."""
    return _REMAP_LUT[mask]


def get_train_transforms(img_size=256):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3),
        A.GaussNoise(p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transforms(img_size=256):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


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
