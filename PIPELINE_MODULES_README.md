# Pipeline Module Structure

This directory contains the SegFormer semantic segmentation pipeline split into separate modules:

## Module Organization

### Setup & Configuration
- **01_imports_setup.py** - Import statements and device configuration
- **02_global_config.py** - Global hyperparameters, paths, and training configuration
- **03_class_id_mapping.py** - Class ID remapping utilities (original → contiguous indices)

### Data Pipeline
- **04_dataset_class.py** - `SegmentationDataset` class for loading images and masks
- **05_data_augmentation.py** - Augmentation strategies for train/validation
- **06_dataloaders.py** - PyTorch DataLoader creation

### Model & Loss
- **07_model_definition.py** - `SegFormerWrapper` model class
- **08_loss_functions.py** - `DiceFocalLoss` combined loss
- **09_metrics.py** - `MulticlassIoU` metric calculation

### Training
- **10_training_loop.py** - `train_one_epoch()` function
- **11_validation_loop.py** - `validate()` function
- **12_training_execution.py** - Full training pipeline with warmup and scheduler

### Inference & Visualization
- **13_visualization_inference.py** - Inference, postprocessing, and visualization functions

### Testing (Optional)
- **14a_test_dataset_loading.py** - Load test dataset
- **14b_test_model_loading.py** - Load trained model
- **14c_test_evaluation.py** - Evaluate on test set
- **14d_test_visualization.py** - Visualize test predictions

## Usage Example

All modules are modular and import from each other. The original `pipeline.ipynb` contains the notebook-based workflow; these files separate each concern for cleaner code organization.

### To integrate into a training script:

```python
import torch
from global_config import NUM_CLASSES, EPOCHS, BATCH_SIZE, ENCODER_LR, DECODER_LR, WEIGHT_DECAY, WARMUP_EPOCHS, CLASS_IDS, SAVE_DIR
from imports_setup import device
from dataloaders import create_dataloaders
from model_definition import SegFormerWrapper
from loss_functions import DiceFocalLoss
from metrics import MulticlassIoU
from training_execution import run_training

# Create dataloaders
train_loader, val_loader = create_dataloaders()

# Initialize model
model = SegFormerWrapper("nvidia/segformer-b1-finetuned-ade-512-512", NUM_CLASSES).to(device)

# Create loss and metric
criterion = DiceFocalLoss(NUM_CLASSES)
metric = MulticlassIoU(NUM_CLASSES)

# Run training
history, best_miou = run_training(
    model, train_loader, val_loader, criterion, metric, device,
    EPOCHS, ENCODER_LR, DECODER_LR, WEIGHT_DECAY, WARMUP_EPOCHS,
    CLASS_IDS, SAVE_DIR
)
```

## Key Metrics

- **Architecture**: SegFormer (MiT-B2)
- **Classes**: 10 (original IDs: 100, 200, 300, 500, 550, 600, 700, 800, 7100, 10000)
- **Input Size**: 512×512
- **Batch Size**: 4
- **Epochs**: 40
- **Encoder LR**: 6e-5
- **Decoder LR**: 6e-4
- **Weight Decay**: 0.01
