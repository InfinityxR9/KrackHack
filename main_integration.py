# Main Integration Script

"""
Complete training pipeline using modular components.
This script demonstrates how to integrate all the separated modules.
"""

import torch
from global_config import (
    NUM_CLASSES, EPOCHS, BATCH_SIZE, ENCODER_LR, DECODER_LR, 
    WEIGHT_DECAY, WARMUP_EPOCHS, CLASS_IDS, SAVE_DIR
)
from imports_setup import device
from dataloaders import create_dataloaders
from model_definition import SegFormerWrapper
from loss_functions import DiceFocalLoss
from metrics import MulticlassIoU
from training_execution import run_training
from visualization_inference import plot_training_curves


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("SegFormer Semantic Segmentation Pipeline")
    print("=" * 60)
    
    # Step 1: Create dataloaders
    print("\n[1/4] Creating dataloaders...")
    train_loader, val_loader = create_dataloaders()
    
    # Step 2: Initialize model
    print("\n[2/4] Initializing model...")
    model = SegFormerWrapper(
        "nvidia/segformer-b1-finetuned-ade-512-512", 
        NUM_CLASSES
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Step 3: Create loss and metric
    print("\n[3/4] Setting up loss and metric...")
    criterion = DiceFocalLoss(NUM_CLASSES)
    metric = MulticlassIoU(NUM_CLASSES)
    
    # Step 4: Run training
    print("\n[4/4] Starting training...")
    history, best_miou = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        metric=metric,
        device=device,
        EPOCHS=EPOCHS,
        ENCODER_LR=ENCODER_LR,
        DECODER_LR=DECODER_LR,
        WEIGHT_DECAY=WEIGHT_DECAY,
        WARMUP_EPOCHS=WARMUP_EPOCHS,
        CLASS_IDS=CLASS_IDS,
        SAVE_DIR=SAVE_DIR
    )
    
    # Plot training curves
    print("\nPlotting training curves...")
    plot_training_curves(history)
    
    print("\n" + "=" * 60)
    print(f"Training complete! Best mIoU: {best_miou:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
