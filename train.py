"""
Training and validation loop for multiclass semantic segmentation.

Saves the best model (by validation mIoU) and prints per-epoch metrics.
Uses mixed precision (AMP) for faster training on modern GPUs.
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from dataset import SegmentationDataset, get_train_transforms, get_val_transforms, CLASS_IDS
from model import build_model
from losses import DiceFocalLoss
from metrics import MulticlassIoU


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
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
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


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

    val_loss = running_loss / len(loader.dataset)
    mean_iou, per_class_iou = metric.compute()
    return val_loss, mean_iou, per_class_iou


def main():
    parser = argparse.ArgumentParser(description="Train segmentation model")
    parser.add_argument("--data_root", type=str, default="../dataset",
                        help="Path to dataset root")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Datasets ---
    train_ds = SegmentationDataset(
        image_dir=os.path.join(args.data_root, "train", "Color_Images"),
        mask_dir=os.path.join(args.data_root, "train", "segmentation"),
        transform=get_train_transforms(args.img_size),
    )
    val_ds = SegmentationDataset(
        image_dir=os.path.join(args.data_root, "val", "Color_Images"),
        mask_dir=os.path.join(args.data_root, "val", "segmentation"),
        transform=get_val_transforms(args.img_size),
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    print(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    # --- Model, Loss, Optimizer ---
    model = build_model().to(device)
    criterion = DiceFocalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler(enabled=(device.type == "cuda"))
    metric = MulticlassIoU()

    # --- Training Loop ---
    best_miou = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_miou, per_class_iou = validate(model, val_loader, criterion, metric, device)

        print(f"Epoch {epoch}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val mIoU: {val_miou:.4f}")

        # Print per-class IoU every 10 epochs for monitoring
        if epoch % 10 == 0 or epoch == 1:
            for cls_idx, iou_val in enumerate(per_class_iou):
                print(f"  Class {CLASS_IDS[cls_idx]:>5d} (idx {cls_idx}): IoU = {iou_val:.4f}")

        # Save best model
        if val_miou > best_miou:
            best_miou = val_miou
            save_path = os.path.join(args.save_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved best model (mIoU={best_miou:.4f})")

    # Save final checkpoint
    torch.save(model.state_dict(), os.path.join(args.save_dir, "final_model.pth"))
    print(f"\nTraining complete. Best mIoU: {best_miou:.4f}")


if __name__ == "__main__":
    main()
