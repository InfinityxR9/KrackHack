# 10. Training Loop

import torch
from torch.amp import autocast


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    print("Training Epoch")
    model.train()
    running_loss = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            preds = model(images)
            loss = criterion(preds, masks)

        scaler.scale(loss).backward()

        # Gradient clipping — critical for transformer stability.
        # Without it, self-attention gradients can spike (especially
        # in early epochs before warmup completes), causing NaN losses.
        # This was NOT needed for the CNN baseline but IS needed here.
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)

    # drop_last=True means we process fewer than len(dataset) samples
    total_samples = len(loader) * loader.batch_size
    return running_loss / total_samples
