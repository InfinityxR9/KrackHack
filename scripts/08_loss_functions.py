# 8. Loss Functions

import torch.nn as nn
import segmentation_models_pytorch as smp


class DiceFocalLoss(nn.Module):
    """Dice + Focal combined loss for multiclass segmentation."""

    def __init__(self, num_classes):
        super().__init__()
        # mode="multiclass" expects predictions [B, C, H, W] and targets [B, H, W] with class indices
        self.dice = smp.losses.DiceLoss(mode="multiclass", classes=num_classes)
        self.focal = smp.losses.FocalLoss(mode="multiclass")

    def forward(self, pred, target):
        return self.dice(pred, target) + self.focal(pred, target)
