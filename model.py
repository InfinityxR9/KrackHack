"""
Model definition: DeepLabV3+ with EfficientNet-B0 encoder.

Uses segmentation-models-pytorch (smp) which provides battle-tested
segmentation architectures with pretrained encoders out of the box.
"""

import segmentation_models_pytorch as smp
from dataset import NUM_CLASSES


def build_model():
    """Build DeepLabV3+ with ImageNet-pretrained EfficientNet-B0 backbone."""
    model = smp.DeepLabV3Plus(
        encoder_name="efficientnet-b0",
        encoder_weights="imagenet",     # pretrained features for faster convergence
        in_channels=3,
        classes=NUM_CLASSES,            # 10 semantic classes
    )
    return model
