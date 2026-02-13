"""
Combined Dice + Focal loss for multiclass semantic segmentation.

Why both?
- Dice Loss directly optimizes the IoU-like overlap metric, handling class imbalance well.
- Focal Loss down-weights easy examples, forcing the model to focus on hard pixels.
- Together they complement each other: Dice for global region overlap, Focal for pixel-level confidence.
"""

import segmentation_models_pytorch as smp
import torch.nn as nn
from dataset import NUM_CLASSES


class DiceFocalLoss(nn.Module):
    """Dice + Focal combined loss for multiclass segmentation."""

    def __init__(self):
        super().__init__()
        # mode="multiclass" expects predictions [B, C, H, W] and targets [B, H, W] with class indices
        self.dice = smp.losses.DiceLoss(mode="multiclass", classes=NUM_CLASSES)
        self.focal = smp.losses.FocalLoss(mode="multiclass")

    def forward(self, pred, target):
        return self.dice(pred, target) + self.focal(pred, target)
