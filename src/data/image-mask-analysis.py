"""
Visualize a NIfTI cardiac MRI slice with its ground-truth segmentation mask overlaid.

Usage:
    python view_nii_overlay.py patient001_frame01.nii patient001_frame01_gt.nii

Or import the functions and call them directly in a notebook.
"""

import numpy as np

import SimpleITK as sitk

import albumentations as A

import matplotlib.pyplot as plt

from utils.data import resize_image
from utils.graphs import show_slice_with_mask

sitk.ProcessObject.SetGlobalWarningDisplay(False)

# ACDC-style labels: 0=background, 1=RV, 2=myocardium, 3=LV
LABEL_NAMES = {0: "background", 1: "RV", 2: "myocardium", 3: "LV"}
LABEL_COLORS = ["none", "#e6194B", "#3cb44b", "#4363d8"]  # transparent bg, red, green, blue


def load_volume(path, is_mask):
    img = sitk.ReadImage(path)
    if not is_mask:
        print("Original size:", img.GetSize())   
    processed_img = resize_image(img, 1.5, is_mask=is_mask)
    if not is_mask:
        print("Processed size:", processed_img.GetSize())   
    img_data = sitk.GetArrayFromImage(processed_img)   # (D, H, W)
    return img_data

def center_crop(image_vol, mask_vol, centre_crop_size):
    train_transform = A.Compose([
        A.PadIfNeeded(min_height=centre_crop_size, min_width=centre_crop_size, fill=0, fill_mask=0),
        A.CenterCrop(height=centre_crop_size, width=centre_crop_size),      # or whatever your model input is
    ])
    aug_images = []
    aug_masks = []
    for image, mask in zip(image_vol, mask_vol):
        cropped = train_transform(image=image, mask=mask)  # applied per-slice, batched over D
        aug_images.append(cropped["image"])
        aug_masks.append(cropped["mask"])

    return aug_images, aug_masks


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


def show_all_slices(image_path, mask_path, alpha=0.4, save_path=None):
    """Grid view of every slice in the volume, image+mask overlaid."""
    image_vol = load_volume(image_path, False)
    mask_vol = load_volume(mask_path, True)

    inspect_labels(mask_vol, path=mask_path)

    n_slices = image_vol.shape[0]
    cols = min(5, n_slices)
    rows = int(np.ceil((n_slices / cols) * 2))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.atleast_1d(axes).flatten()

    for i in range(n_slices):
        show_slice_with_mask(image_vol[i], mask_vol[i], alpha=alpha, ax=axes[i], title=f"Slice {i}")

    image_vol, mask_vol = center_crop(image_vol, mask_vol, centre_crop_size=200)

    for i in range(n_slices):
            show_slice_with_mask(image_vol[i], mask_vol[i], alpha=alpha, ax=axes[i + n_slices], title=f"Slice augmented {i}")

    for i in range(n_slices * 2, len(axes)):
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
    # largest patient voxel = 135, lowest = 85
    patient_num = 57
    path = f"data/raw/training/patient{patient_num:03d}/"
    image_path = path + f"patient{patient_num:03d}_frame01.nii"
    mask_path = path + f"patient{patient_num:03d}_frame01_gt.nii"
    show_all_slices(image_path, mask_path, save_path="output/analysis/overlay_grid.png")