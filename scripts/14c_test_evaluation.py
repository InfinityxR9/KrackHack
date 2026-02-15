# 14-C: Test Evaluation

import numpy as np
import torch
from tqdm import tqdm


def evaluate_on_test(model, loader, num_classes, device):
    """Evaluate the model on the test dataset and compute per-class and mean IoU."""
    model.eval()

    intersection = np.zeros(num_classes)
    union = np.zeros(num_classes)
    gt_pixels = np.zeros(num_classes)

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Evaluating TEST dataset"):
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            preds = torch.argmax(logits, dim=1)

            for cls in range(num_classes):
                pred_c = (preds == cls)
                gt_c   = (masks == cls)

                intersection[cls] += torch.logical_and(pred_c, gt_c).sum().item()
                union[cls] += torch.logical_or(pred_c, gt_c).sum().item()
                gt_pixels[cls] += gt_c.sum().item()

    iou_per_class = []
    for cls in range(num_classes):
        if gt_pixels[cls] == 0:
            iou_per_class.append(None)  # class not present in GT
        else:
            iou_per_class.append(intersection[cls] / (union[cls] + 1e-7))

    valid_ious = [iou for iou in iou_per_class if iou is not None]
    miou = np.mean(valid_ious)

    return iou_per_class, miou
