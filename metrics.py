"""
Multiclass IoU (Intersection over Union) metric, also known as Jaccard Index.

Computes per-class IoU and returns the mean across all classes.
This is the standard metric for semantic segmentation evaluation.
"""

import torch
import numpy as np
from dataset import NUM_CLASSES


class MulticlassIoU:
    """Accumulates predictions over batches, then computes mean IoU."""

    def __init__(self, num_classes=NUM_CLASSES):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        """Clear accumulators at the start of each epoch."""
        self.intersection = np.zeros(self.num_classes)
        self.union = np.zeros(self.num_classes)

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Args:
            pred: raw logits [B, C, H, W]
            target: class indices [B, H, W]
        """
        # Convert logits to predicted class indices
        pred_classes = pred.argmax(dim=1).cpu().numpy()  # [B, H, W]
        target_np = target.cpu().numpy()                  # [B, H, W]

        for cls in range(self.num_classes):
            pred_mask = (pred_classes == cls)
            target_mask = (target_np == cls)
            self.intersection[cls] += np.logical_and(pred_mask, target_mask).sum()
            self.union[cls] += np.logical_or(pred_mask, target_mask).sum()

    def compute(self):
        """Return mean IoU across all classes. Ignores classes with zero union."""
        iou_per_class = np.zeros(self.num_classes)
        for cls in range(self.num_classes):
            if self.union[cls] > 0:
                iou_per_class[cls] = self.intersection[cls] / self.union[cls]
        mean_iou = iou_per_class.mean()
        return mean_iou, iou_per_class
