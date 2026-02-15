# 6. DataLoaders

import os
from torch.utils.data import DataLoader
from dataset_class import SegmentationDataset
from data_augmentation import get_train_transforms, get_val_transforms
from global_config import DATA_ROOT, BATCH_SIZE, IMG_SIZE, NUM_WORKERS


def create_dataloaders():
    """Create train and validation dataloaders."""
    train_ds = SegmentationDataset(
        image_dir=os.path.join(DATA_ROOT, "train", "Color_Images"),
        mask_dir=os.path.join(DATA_ROOT, "train", "segmentation"),
        transform=get_train_transforms(IMG_SIZE),
    )
    val_ds = SegmentationDataset(
        image_dir=os.path.join(DATA_ROOT, "val", "Color_Images"),
        mask_dir=os.path.join(DATA_ROOT, "val", "segmentation"),
        transform=get_val_transforms(IMG_SIZE),
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
    )

    print(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")
    
    return train_loader, val_loader
