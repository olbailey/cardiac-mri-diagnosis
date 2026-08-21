import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import torch
from torch.utils.data import DataLoader


LABEL_NAMES = {0: "background", 1: "RV", 2: "myocardium", 3: "LV"}
LABEL_COLORS = ["none", "#e6194B", "#3cb44b", "#4363d8"]

def show_slice_with_mask(img_slice, mask_slice, title, alpha=0.4, ax=None, hide_axis=False):
    """Display one slice of a 3D volume with its mask overlaid in color.

    NIfTI arrays are stored (row, col) which often doesn't match how you'd
    expect the image to look on screen, and radiological convention can add
    a flip on top of that. Defaults here (transpose + origin='lower') work
    for most ACDC-style data, but ALWAYS check visually against a reference
    viewer (e.g. ITK-SNAP, or the .nii metadata) before trusting orientation
    for training. If the heart looks sideways or mirrored, toggle these:
        transpose : swap rows/cols (usually needed, default True)
        flip_ud   : flip vertically after transpose
        flip_lr   : flip horizontally after transpose
    """

    # normalize image for display
    img_norm = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-8)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    ax.imshow(img_norm, cmap="gray", origin="lower")

    cmap = ListedColormap(LABEL_COLORS)
    masked = np.ma.masked_where(mask_slice == 0, mask_slice)  # don't paint background
    ax.imshow(masked, cmap=cmap, vmin=0, vmax=3, alpha=alpha, origin="lower")

    ax.set_title(title)

    if hide_axis:
        ax.axis("off")

    return ax

def show_single_slice_prediction(model: torch.nn.Module, test_dataset, slice_name, classes_num, device, alpha=0.4):
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=LABEL_COLORS[i])
        for i in [1, 2, 3]
    ]
    plt.legend(handles, [LABEL_NAMES[i] for i in [1, 2, 3]], loc="upper right")

    model.eval()
    with torch.no_grad():
        img_slice, mask_slice, plain_slice = test_dataset.apply_transform(slice_name)

        output = model(img_slice.to(device).unsqueeze(0))
        predict = output.squeeze(0).cpu().numpy()
        predict = np.argmax(predict, axis=0)

        img_slice = img_slice.squeeze().cpu().numpy()

        img_norm = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-8)
        plt.imshow(img_norm, cmap="gray", origin="lower")

        cmap = ListedColormap(LABEL_COLORS)
        masked = np.ma.masked_where(predict == 0, predict)  # don't paint background
        plt.imshow(masked, cmap=cmap, vmin=0, vmax=3, alpha=alpha, origin="lower")

        plt.show()

# def plot_predictions(model, loader: DataLoader, device: torch.device, num_points: int = None, title: str = "Predicted vs Actual"):
#     """
#     Runs the model over a DataLoader and plots predicted vs actual values.

#     num_points: if set, only plots the first N points (useful for zooming in
#                 on long time series where plotting everything is unreadable).
#     """
#     model.eval()
#     all_preds = []
#     all_targets = []

#     with torch.no_grad():
#         for inputs, targets in loader:
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs = model(inputs).squeeze(-1)

#             all_preds.append(outputs.cpu())
#             all_targets.append(targets.cpu())

#     preds = torch.cat(all_preds).numpy()
#     targets = torch.cat(all_targets).numpy()

#     if num_points is not None and num_points > 0:
#         preds = preds[:num_points]
#         targets = targets[:num_points]

#     fig, axes = plt.subplots(2, 1, figsize=(12, 8))

#     # Top plot: predicted vs actual over "time" (i.e. sample index)
#     axes[0].plot(targets, label='Actual', linewidth=1.5, alpha=0.8)
#     axes[0].plot(preds, label='Predicted', linewidth=1.5, alpha=0.8)
#     axes[0].set_xlabel('Sample index')
#     axes[0].set_ylabel('Value')
#     axes[0].set_title(title)
#     axes[0].legend()
#     axes[0].grid(alpha=0.3)

#     # Bottom plot: scatter of predicted vs actual (perfect predictions fall on y=x line)
#     axes[1].scatter(targets, preds, alpha=0.4, s=10)
#     min_val = min(targets.min(), preds.min())
#     max_val = max(targets.max(), preds.max())
#     axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1, label='Perfect prediction')
#     axes[1].set_xlabel('Actual')
#     axes[1].set_ylabel('Predicted')
#     axes[1].set_title('Predicted vs Actual (scatter)')
#     axes[1].legend()
#     axes[1].grid(alpha=0.3)

#     plt.tight_layout()
#     plt.show()

#     return preds, targets