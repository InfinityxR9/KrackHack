# 7. Model Definition

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation


class SegFormerWrapper(nn.Module):
    """
    Wraps HuggingFace SegformerForSemanticSegmentation to output logits
    at full input resolution [B, NUM_CLASSES, H, W].

    SegFormer natively outputs at H/4 x W/4. We bilinear-upsample so
    the existing loss (DiceFocalLoss) and metric (MulticlassIoU) code
    works without any modification.
    """

    def __init__(self, pretrained_name, num_classes):
        super().__init__()
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,  # reinit classifier for our 10 classes
        )

    def forward(self, x):
        outputs = self.model(pixel_values=x)
        logits = outputs.logits  # [B, num_classes, H/4, W/4]
        # Upsample to input resolution for loss/metric compatibility
        logits = F.interpolate(
            logits, size=x.shape[2:], mode="bilinear", align_corners=False
        )
        return logits
