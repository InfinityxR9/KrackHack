# 12. Training Execution

import os
import math
import torch
from torch.cuda.amp import GradScaler


def run_training(model, train_loader, val_loader, criterion, metric, device,
                 EPOCHS, ENCODER_LR, DECODER_LR, WEIGHT_DECAY, WARMUP_EPOCHS, 
                 CLASS_IDS, SAVE_DIR):
    """
    Runs the full training loop with differential learning rates and warmup.
    
    Key differences from DeepLabV3+ training:
    - **Differential LR**: encoder (MiT-B2) at 6e-5, decode head at 6e-4
    - **AdamW** with weight_decay=0.01 (standard for transformers)
    - **Linear warmup** (3 epochs) + cosine decay — warmup is critical for transformers
    - **Scheduler actually stepped** (the baseline defined CosineAnnealingLR but never called .step())
    """
    
    # --- Differential learning rates for transformer fine-tuning ---
    optimizer = torch.optim.AdamW([
        {"params": model.model.segformer.parameters(), "lr": ENCODER_LR},
        {"params": model.model.decode_head.parameters(), "lr": DECODER_LR},
    ], weight_decay=WEIGHT_DECAY)

    # --- Warmup + cosine decay scheduler ---
    def lr_lambda_fn(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS  # linear: 0.33 -> 0.67 -> 1.0
        # Cosine decay over remaining epochs
        progress = (epoch - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda_fn)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    best_miou = 0.0
    history = {"train_loss": [], "val_loss": [], "val_miou": []}

    from train_loop import train_one_epoch
    from validation_loop import validate

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_miou, per_class_iou = validate(model, val_loader, criterion, metric, device)
        scheduler.step()  # Step per epoch (was missing in baseline!)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_miou"].append(val_miou)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val mIoU: {val_miou:.4f} | "
              f"LR: {current_lr:.2e}")

        # Print per-class IoU every 10 epochs for monitoring
        if epoch % 10 == 0 or epoch == 1:
            for cls_idx, iou_val in enumerate(per_class_iou):
                print(f"  Class {CLASS_IDS[cls_idx]:>5d} (idx {cls_idx}): IoU = {iou_val:.4f}")

        # Save best model
        if val_miou > best_miou:
            best_miou = val_miou
            save_path = os.path.join(SAVE_DIR, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved best model (mIoU={best_miou:.4f})")

    # Save final checkpoint
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "final_model.pth"))
    print(f"\nTraining complete. Best mIoU: {best_miou:.4f}")
    
    return history, best_miou
