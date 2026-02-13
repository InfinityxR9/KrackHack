"""
Visualization: overlay predicted segmentation masks on original images.
Useful for debugging, qualitative evaluation, and hackathon presentations.
"""

import os
import argparse
import cv2
import numpy as np
from dataset import CLASS_IDS, NUM_CLASSES

# Distinct colors for each class (BGR format for OpenCV)
CLASS_COLORS = [
    (0,   0, 128),    # 100  - dark red
    (0, 128,   0),    # 200  - green
    (128, 0,   0),    # 300  - dark blue
    (0, 128, 128),    # 500  - yellow-ish
    (128, 128, 0),    # 550  - cyan-ish
    (128, 0, 128),    # 600  - magenta
    (0, 255,   0),    # 700  - bright green
    (255, 0,   0),    # 800  - blue
    (0, 255, 255),    # 7100 - yellow
    (255, 255, 0),    # 10000- cyan
]

CLASS_NAMES = [
    f"cls_{cid}" for cid in CLASS_IDS
]


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Convert class-index mask to a colored RGB image."""
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx in range(NUM_CLASSES):
        colored[mask == cls_idx] = CLASS_COLORS[cls_idx]
    return colored


def overlay(image: np.ndarray, mask: np.ndarray, alpha=0.5) -> np.ndarray:
    """Blend colored mask onto the original image."""
    colored = colorize_mask(mask)
    blended = cv2.addWeighted(image, 1 - alpha, colored, alpha, 0)
    return blended


def add_legend(image: np.ndarray) -> np.ndarray:
    """Add a class-color legend strip to the right side of the image."""
    h = image.shape[0]
    legend_w = 160
    legend = np.ones((h, legend_w, 3), dtype=np.uint8) * 255  # white background

    box_h = h // NUM_CLASSES
    for idx in range(NUM_CLASSES):
        y_start = idx * box_h
        # Color swatch
        cv2.rectangle(legend, (5, y_start + 4), (25, y_start + box_h - 4),
                       CLASS_COLORS[idx], -1)
        # Label text
        cv2.putText(legend, f"{CLASS_IDS[idx]}", (32, y_start + box_h - 8),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    return np.hstack([image, legend])


def main():
    parser = argparse.ArgumentParser(description="Visualize segmentation results")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory of original images")
    parser.add_argument("--mask_dir", type=str, required=True,
                        help="Directory of predicted masks (contiguous class indices)")
    parser.add_argument("--output_dir", type=str, default="visualizations")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Overlay transparency")
    parser.add_argument("--max_images", type=int, default=20,
                        help="Max number of images to visualize")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    image_files = sorted([f for f in os.listdir(args.image_dir) if f.endswith(".png")])
    mask_files = sorted([f for f in os.listdir(args.mask_dir) if f.endswith(".png")])

    # Visualize using filename matching (both dirs should have same filenames from infer.py)
    common = [f for f in image_files if f in mask_files]
    if not common:
        # Fall back to index-based pairing
        common = list(zip(image_files, mask_files))
        use_pairs = True
    else:
        use_pairs = False

    count = 0
    items = common[:args.max_images]

    for item in items:
        if use_pairs:
            img_name, mask_name = item
        else:
            img_name = mask_name = item

        image = cv2.imread(os.path.join(args.image_dir, img_name), cv2.IMREAD_COLOR)
        mask = cv2.imread(os.path.join(args.mask_dir, mask_name), cv2.IMREAD_UNCHANGED)

        if mask.ndim == 3:
            mask = mask[:, :, 0]

        # Resize mask to image size if needed
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

        # Create overlay
        vis = overlay(image, mask, alpha=args.alpha)
        vis = add_legend(vis)

        out_path = os.path.join(args.output_dir, f"vis_{img_name}")
        cv2.imwrite(out_path, vis)
        count += 1

    print(f"Saved {count} visualizations to {args.output_dir}/")


if __name__ == "__main__":
    main()
