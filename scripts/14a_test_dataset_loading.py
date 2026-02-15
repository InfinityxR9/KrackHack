# 14-A: Test Dataset Loading

import os
from torch.utils.data import DataLoader
from dataset_class import SegmentationDataset
from data_augmentation import get_val_transforms
from global_config import BATCH_SIZE, IMG_SIZE, NUM_WORKERS


def create_test_loader(test_data_root="testing_dataset"):
    """Create a dataloader for the test dataset."""
    test_image_dir = f"{test_data_root}/Color_Images"
    test_mask_dir  = f"{test_data_root}/segmentation"

    val_transform = get_val_transforms(img_size=IMG_SIZE)

    test_dataset = SegmentationDataset(
        image_dir=test_image_dir,
        mask_dir=test_mask_dir,
        transform=val_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    print(f"Loaded TEST dataset with {len(test_dataset)} samples")
    
    return test_loader, test_dataset
