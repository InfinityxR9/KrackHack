# 13. Visualization & Inference

import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.amp import autocast


# Distinct colors for each class (RGB for matplotlib)
CLASS_COLORS = [
    (128,   0,   0),    # 100  - dark red
    (  0, 128,   0),    # 200  - green
    (  0,   0, 128),    # 300  - dark blue
    (128, 128,   0),    # 500  - yellow-ish
    (  0, 128, 128),    # 550  - cyan-ish
    (128,   0, 128),    # 600  - magenta
    (  0, 255,   0),    # 700  - bright green
    (  0,   0, 255),    # 800  - blue
    (255, 255,   0),    # 7100 - yellow
    (  0, 255, 255),    # 10000- cyan
]


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Convert class-index mask to a colored RGB image."""
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx in range(len(CLASS_COLORS)):
        colored[mask == cls_idx] = CLASS_COLORS[cls_idx]
    return colored


def postprocess_mask(mask: np.ndarray) -> np.ndarray:
    """
    Morphological post-processing to clean up noisy predictions.
    Applies per-class opening (remove small spurious pixels)
    followed by closing (fill small holes).
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = np.zeros_like(mask)
    num_classes = len(CLASS_COLORS)

    for cls in range(num_classes):
        binary = (mask == cls).astype(np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        cleaned[binary == 1] = cls

    return cleaned


def remap_to_original(mask: np.ndarray, class_ids) -> np.ndarray:
    """Convert contiguous labels [0-9] back to original class IDs for submission."""
    out = np.zeros_like(mask, dtype=np.uint16)
    for idx, original_id in enumerate(class_ids):
        out[mask == idx] = original_id
    return out


def visualize_inference(model, val_image_dir, val_mask_dir, val_images, val_masks, 
                        infer_transform, device, CLASS_IDS, NUM_SAMPLES=6, IMG_SIZE=512):
    """
    Visual inference on validation samples.
    """
    model.eval()

    fig, axes = plt.subplots(NUM_SAMPLES, 3, figsize=(15, 5 * NUM_SAMPLES))
    if NUM_SAMPLES == 1:
        axes = axes[np.newaxis, :]  # ensure 2D indexing

    from class_id_mapping import remap_mask

    for i in range(NUM_SAMPLES):
        # Load original image
        img_path = os.path.join(val_image_dir, val_images[i])
        image_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        original_h, original_w = image_rgb.shape[:2]

        # Load ground truth mask
        gt_path = os.path.join(val_mask_dir, val_masks[i])
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
        if gt_mask.ndim == 3:
            gt_mask = gt_mask[:, :, 0]
        gt_mask = remap_mask(gt_mask).astype(np.uint8)

        # Run inference
        augmented = infer_transform(image=image_rgb)
        input_tensor = augmented["image"].unsqueeze(0).to(device)

        with torch.no_grad(), autocast(device_type="cuda", enabled=(device.type == "cuda")):
            logits = model(input_tensor)

        pred_mask = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        # Resize prediction back to original image dimensions
        pred_mask = cv2.resize(pred_mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

        # Plot: original | ground truth | prediction
        axes[i, 0].imshow(image_rgb)
        axes[i, 0].set_title(f"Image: {val_images[i]}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(colorize_mask(gt_mask))
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(colorize_mask(pred_mask))
        axes[i, 2].set_title("Prediction")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig("inference_results.png", dpi=150)
    plt.show()


def plot_training_curves(history):
    """Plot training loss, validation loss, and mIoU."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history["val_miou"], label="Val mIoU", color="green")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("mIoU")
    axes[1].set_title("Validation mIoU")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.show()
