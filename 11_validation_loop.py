# 11. Validation Loop

import torch
from torch.amp import autocast


@torch.no_grad()
def validate(model, loader, criterion, metric, device):
    model.eval()
    running_loss = 0.0
    metric.reset()

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            preds = model(images)
            loss = criterion(preds, masks)

        running_loss += loss.item() * images.size(0)
        metric.update(preds, masks)

    # Same drop_last correction as training
    total_samples = len(loader) * loader.batch_size
    val_loss = running_loss / total_samples
    mean_iou, per_class_iou = metric.compute()
    return val_loss, mean_iou, per_class_iou
