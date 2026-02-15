# 14-D: Test Visualization

import numpy as np
import torch
import matplotlib.pyplot as plt


def visualize_test_predictions(model, dataset, device, num_samples=6):
    """Visualize test predictions on random samples."""
    model.eval()
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    for idx in indices:
        image, gt_mask = dataset[idx]
        image_tensor = image.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(image_tensor)
            pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu()

        img_np = image.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        ax[0].imshow(img_np)
        ax[0].set_title("Input Image")

        ax[1].imshow(gt_mask, cmap="tab20")
        ax[1].set_title("Ground Truth")

        ax[2].imshow(pred_mask, cmap="tab20")
        ax[2].set_title("Prediction")

        for a in ax:
            a.axis("off")

        plt.show()
