# 5. Data Augmentation

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(img_size=512):
    """
    Augmentation strategy adapted for SegFormer (transformer architecture).

    vs. CNN (DeepLabV3+) pipeline, three principles guide the changes:

    1. GEOMETRIC AUGMENTATIONS RETAINED at moderate intensity.
       Transformers lack CNN's built-in translation equivariance.
       They must learn spatial invariance from data, so geometric
       diversity (shifts, scales, rotations) remains critical.

    2. COLOR / NOISE AUGMENTATIONS REDUCED.
       Transformers tokenize images into patches and embed them linearly.
       Heavy color distortion destabilizes patch embeddings more than
       conv features, especially early in fine-tuning. Self-attention
       already generalizes texture better than CNNs, so heavy noise
       provides diminishing returns.

    3. ImageCompression REMOVED.
       Broken API in albumentations v2 + minimal benefit for transformers.
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        # Geometric: kept similar to baseline (shift/scale/rotate).
        # Slightly reduced rotate_limit (30->20) — less extreme rotations
        # are more realistic for off-road camera views with stable horizon.
        A.ShiftScaleRotate(
            shift_limit=0.1, scale_limit=0.15, rotate_limit=20,
            border_mode=cv2.BORDER_REFLECT_101, p=0.5,
        ),
        # Desert lighting: harsh shadows are a real domain trait. Kept.
        A.RandomShadow(p=0.3),
        # REDUCED from baseline (0.3/0.3 limits, p=0.6 -> 0.2/0.2, p=0.4).
        # Protects pretrained patch embeddings from too much brightness shock.
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        # REDUCED from baseline (0.2/0.2/0.2/0.1 at p=0.5 -> 0.1/0.1/0.1/0.05 at p=0.25).
        # Mild jitter preserves embedding stability while still adding
        # color variance for synthetic-to-real domain gap.
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.25),
        # REDUCED overall probability (p=0.4 -> 0.2), removed ImageCompression.
        # Transformers generalize texture through self-attention across patches,
        # so heavy noise/blur just hurts convergence without useful regularization.
        A.OneOf([
            A.GaussNoise(p=1.0),
            A.GaussianBlur(blur_limit=3, p=1.0),
        ], p=0.2),
        # ImageNet normalization — non-negotiable. SegFormer's MiT encoder
        # was pretrained with these exact statistics.
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transforms(img_size=512):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
