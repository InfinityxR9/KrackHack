# 2. Global Configuration

import os

DATA_ROOT = "dataset"
SAVE_DIR = "checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

# Training hyperparameters
EPOCHS = 40           # Increased from 30: transformers converge slower
BATCH_SIZE = 4        # Same as DeepLabV3+ baseline — fits 8GB VRAM with AMP
IMG_SIZE = 512
NUM_WORKERS = 0

# SegFormer-specific configuration
# B2 pretrained on ADE20K (150 scene classes) — strong transfer for outdoor scenes.
# The classifier head gets reinitialized for our 10 classes.
# SEGFORMER_VARIANT = "nvidia/segformer-b2-finetuned-ade-512-512"
# If OOM on your GPU, uncomment the B1 variant (~14M params vs ~27M for B2):
SEGFORMER_VARIANT = "nvidia/segformer-b1-finetuned-ade-512-512"

# Learning rates — transformers need lower LR than CNNs to avoid
# destabilizing pretrained self-attention weights.
ENCODER_LR = 6e-5    # Gentle: preserve pretrained MiT representations
DECODER_LR = 6e-4    # Faster: MLP decode head adapts to our classes
WEIGHT_DECAY = 0.01  # Standard for transformers (vs 1e-4 for CNNs)
WARMUP_EPOCHS = 3    # Linear warmup before cosine decay
