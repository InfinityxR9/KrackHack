# Test Integration Script

"""
Complete testing pipeline using modular components.
This script demonstrates how to integrate all the test-related modules.
"""

import torch
from imports_setup import device
from global_config import NUM_CLASSES, CLASS_IDS
from test_dataset_loading import create_test_loader
from test_model_loading import load_model_for_testing
from test_evaluation import evaluate_on_test
from test_visualization import visualize_test_predictions


def main():
    """Main testing pipeline."""
    print("=" * 60)
    print("SegFormer Semantic Segmentation - Testing Pipeline")
    print("=" * 60)
    
    # Step 1: Load test dataset
    print("\n[1/3] Loading test dataset...")
    test_loader, test_dataset = create_test_loader("testing_dataset")
    
    # Step 2: Load trained model
    print("\n[2/3] Loading trained model...")
    model = load_model_for_testing(device, "checkpoints/best_model.pth")
    
    # Step 3: Evaluate on test set
    print("\n[3/3] Evaluating on test set...")
    iou_per_class, test_miou = evaluate_on_test(
        model, test_loader, NUM_CLASSES, device
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("TEST DATASET RESULTS")
    print("=" * 60)
    
    for idx, iou in enumerate(iou_per_class):
        cls_id = CLASS_IDS[idx]
        if iou is None:
            print(f"Class {cls_id:>5d}: NOT PRESENT in GT")
        else:
            print(f"Class {cls_id:>5d}: IoU = {iou:.4f}")

    print("-" * 60)
    print(f"Overall TEST mIoU (valid classes only): {test_miou:.4f}")
    print("=" * 60)
    
    # Visualize predictions
    print("\nVisualizing test predictions...")
    visualize_test_predictions(model, test_dataset, device, num_samples=6)


if __name__ == "__main__":
    main()
