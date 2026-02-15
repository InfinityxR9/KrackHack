# 14-B: Test Model Loading

import os
import torch
from model_definition import SegFormerWrapper
from global_config import SEGFORMER_VARIANT, NUM_CLASSES


def load_model_for_testing(device, model_path="checkpoints/best_model.pth"):
    """Load the best trained model for testing."""
    model = SegFormerWrapper(SEGFORMER_VARIANT, NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    print("Loaded best SegFormer model for testing")
    
    return model
