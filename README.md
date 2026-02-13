# Off-Road Desert Semantic Segmentation

Multiclass semantic segmentation for off-road desert environments using DeepLabV3+ with EfficientNet-B0.

## Setup

```bash
pip install -r requirements.txt
```

## Dataset

Expected structure at `../dataset/` (relative to this project folder):

```
dataset/
  train/
    Color_Images/   (2857 images)
    segmentation/   (2857 masks)
  val/
    Color_Images/   (317 images)
    segmentation/   (317 masks)
```

Images and masks are paired by **sorted index**, not by filename.

Masks are uint16 PNGs with class IDs: `{100, 200, 300, 500, 550, 600, 700, 800, 7100, 10000}`, remapped internally to `[0-9]`.

## Train

```bash
python train.py --data_root ../dataset --epochs 50 --batch_size 8
```

Key flags:
- `--lr 1e-4` (default)
- `--img_size 256` (default)
- `--save_dir checkpoints` (default)

## Inference

```bash
python infer.py --input_dir ../dataset/val/Color_Images --checkpoint checkpoints/best_model.pth --postprocess
```

Add `--remap` to output original class IDs instead of contiguous [0-9].

## Visualize

```bash
python visualize.py --image_dir ../dataset/val/Color_Images --mask_dir predictions
```

## Architecture

- **Model**: DeepLabV3+ (segmentation-models-pytorch)
- **Encoder**: EfficientNet-B0 (ImageNet pretrained)
- **Loss**: Dice + Focal
- **Metric**: Mean IoU
- **Optimizer**: Adam (lr=1e-4)
- **Augmentations**: HorizontalFlip, RandomBrightnessContrast, ColorJitter, GaussNoise
