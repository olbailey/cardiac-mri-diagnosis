"""
Visualize a NIfTI cardiac MRI slice with its ground-truth segmentation mask overlaid.

Usage:
    python view_nii_overlay.py patient001_frame01.nii patient001_frame01_gt.nii

Or import the functions and call them directly in a notebook.
"""

import sys
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# ACDC-style labels: 0=background, 1=RV, 2=myocardium, 3=LV
LABEL_NAMES = {0: "background", 1: "RV", 2: "myocardium", 3: "LV"}
LABEL_COLORS = ["none", "#e6194B", "#3cb44b", "#4363d8"]  # transparent bg, red, green, blue


def load_volume(path):
    img = nib.load(path)
    return img.get_fdata()


def inspect_labels(mask_vol, path=""):
    """Print the unique integer values present in a mask volume.

    Run this before trusting LABEL_NAMES/LABEL_COLORS above. Datasets vary:
    some use 0/1/2/3, others number differently or include extra structures.
    If the printed values don't match LABEL_NAMES, update LABEL_NAMES and
    LABEL_COLORS accordingly before visualizing.
    """
    values = np.unique(mask_vol)
    print(f"Unique label values in {path or 'mask'}: {values}")
    for v in values:
        name = LABEL_NAMES.get(int(v), "UNKNOWN — update LABEL_NAMES")
        print(f"  {int(v)}: {name}")
    return values


def show_slice_with_mask(image_vol, mask_vol, slice_idx=None, alpha=0.4, ax=None,
                          transpose=True, flip_ud=False, flip_lr=False):
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
    if slice_idx is None:
        slice_idx = image_vol.shape[2] // 2  # middle slice by default

    img_slice = image_vol[:, :, slice_idx]
    mask_slice = mask_vol[:, :, slice_idx]

    if transpose:
        img_slice = img_slice.T
        mask_slice = mask_slice.T
    if flip_ud:
        img_slice = np.flipud(img_slice)
        mask_slice = np.flipud(mask_slice)
    if flip_lr:
        img_slice = np.fliplr(img_slice)
        mask_slice = np.fliplr(mask_slice)

    # normalize image for display
    img_norm = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-8)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    ax.imshow(img_norm, cmap="gray", origin="lower")

    cmap = ListedColormap(LABEL_COLORS)
    masked = np.ma.masked_where(mask_slice == 0, mask_slice)  # don't paint background
    ax.imshow(masked, cmap=cmap, vmin=0, vmax=3, alpha=alpha, origin="lower")

    ax.set_title(f"Slice {slice_idx}")
    ax.axis("off")
    return ax


def show_all_slices(image_path, mask_path, alpha=0.4, save_path=None,
                     transpose=True, flip_ud=False, flip_lr=False):
    """Grid view of every slice in the volume, image+mask overlaid."""
    image_vol = load_volume(image_path)
    mask_vol = load_volume(mask_path)
    # print(mask_vol.shape)
    # arr = np.array(mask_vol)
    # arr = arr.reshape((10, 216, 256))
    # print(arr.shape)
    # print(np.unique(arr))

    # return

    inspect_labels(mask_vol, path=mask_path)

    n_slices = image_vol.shape[2]
    cols = min(5, n_slices)
    rows = int(np.ceil(n_slices / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.atleast_1d(axes).flatten()

    for i in range(n_slices):
        show_slice_with_mask(image_vol, mask_vol, slice_idx=i, alpha=alpha, ax=axes[i],
                              transpose=transpose, flip_ud=flip_ud, flip_lr=flip_lr)

    for i in range(n_slices, len(axes)):
        axes[i].axis("off")

    # legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=LABEL_COLORS[i])
        for i in [1, 2, 3]
    ]
    fig.legend(handles, [LABEL_NAMES[i] for i in [1, 2, 3]], loc="upper right")

    fig.suptitle(f"{image_path.split('/')[-1]}  (n_slices={n_slices})")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    path = "data/raw/training/patient061/"
    image_path = path + "patient061_frame01.nii"
    mask_path = path + "patient061_frame01_gt.nii"
    show_all_slices(image_path, mask_path, save_path="output/analysis/overlay_grid.png",
                    )