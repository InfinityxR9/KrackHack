"""
Inference script: loads a trained model, runs prediction on images,
and applies OpenCV post-processing (morphological cleanup).

Outputs:
- Raw predicted masks (class index PNGs)
- Post-processed masks (morphologically cleaned)
"""

import os
import argparse
import cv2
import numpy as np
import torch
from torch.cuda.amp import autocast
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import build_model
from dataset import CLASS_IDS, NUM_CLASSES


def get_infer_transform(img_size=256):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def postprocess_mask(mask: np.ndarray) -> np.ndarray:
    """
    Morphological post-processing to clean up noisy predictions.
    Applies per-class opening (remove small spurious pixels)
    followed by closing (fill small holes).
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = np.zeros_like(mask)

    for cls in range(NUM_CLASSES):
        binary = (mask == cls).astype(np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        cleaned[binary == 1] = cls

    return cleaned


def remap_to_original(mask: np.ndarray) -> np.ndarray:
    """Convert contiguous labels [0-9] back to original class IDs for submission."""
    out = np.zeros_like(mask, dtype=np.uint16)
    for idx, original_id in enumerate(CLASS_IDS):
        out[mask == idx] = original_id
    return out


def main():
    parser = argparse.ArgumentParser(description="Run inference on images")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory of input images")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth",
                        help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="predictions")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--postprocess", action="store_true",
                        help="Apply morphological post-processing")
    parser.add_argument("--remap", action="store_true",
                        help="Remap predictions back to original class IDs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = build_model().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    transform = get_infer_transform(args.img_size)
    image_files = sorted([f for f in os.listdir(args.input_dir) if f.endswith(".png")])

    for fname in image_files:
        # Load and preprocess
        img_path = os.path.join(args.input_dir, fname)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        original_h, original_w = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        augmented = transform(image=image_rgb)
        input_tensor = augmented["image"].unsqueeze(0).to(device)  # [1, 3, H, W]

        # Predict
        with torch.no_grad(), autocast(device_type="cuda", enabled=(device.type == "cuda")):
            logits = model(input_tensor)  # [1, NUM_CLASSES, H, W]

        pred_mask = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # Resize prediction back to original image dimensions
        pred_mask = cv2.resize(pred_mask, (original_w, original_h),
                               interpolation=cv2.INTER_NEAREST)

        # Optional post-processing
        if args.postprocess:
            pred_mask = postprocess_mask(pred_mask)

        # Optional remap to original class IDs
        if args.remap:
            save_mask = remap_to_original(pred_mask)
        else:
            save_mask = pred_mask

        # Save prediction
        out_path = os.path.join(args.output_dir, fname)
        cv2.imwrite(out_path, save_mask)

    print(f"Saved {len(image_files)} predictions to {args.output_dir}/")


if __name__ == "__main__":
    main()
