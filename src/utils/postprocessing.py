import os

import numpy as np
import torch
import torch.nn.functional as F

from tqdm.auto import tqdm

PADDING_SIZE = 128

def find_bounding_box(segmentation: torch.Tensor, margin: int):
    # returns the indexes and columns that contain a non zero value
    rows, cols = np.nonzero(segmentation.numpy())

    if rows.size == 0:
        return None
    
    height, width = segmentation.shape
    return {
        "row_min": max(int(rows.min()) - margin, 0),
        "row_max": min(int(rows.max()) + margin + 1, height),  # +1: exclusive end
        "col_min": max(int(cols.min()) - margin, 0),
        "col_max": min(int(cols.max()) + margin + 1, width),
    }

def apply_bounding_box(tensor: torch.Tensor, box, padding=True):
    if padding:
        return pad_to(tensor[box["row_min"]:box["row_max"], box["col_min"]:box["col_max"]]).clone()
    else:
        return tensor[box["row_min"]:box["row_max"], box["col_min"]:box["col_max"]].clone()


def pad_to(arr, size=PADDING_SIZE):
    h, w = arr.shape[-2:]
    if h > size or w > size:
        raise ValueError(f"Crop of {h}x{w} does not fit in a {size}x{size} canvas.")

    pad_h, pad_w = size - h, size - w
    top, left = pad_h // 2, pad_w // 2
    
    return F.pad(arr, (left, pad_w - left, top, pad_h - top), mode="constant", value=0)

def remove_outlier_data(bounding_box) -> bool:
    """For filtering outlier data slices that are not applicable for classification

    Args:
        bounding_box (dict): start and end points of width and height

    Returns:
        bool: whether the slice is invalid
    """

    if bounding_box is None:
        return True

    height = bounding_box["row_max"] - bounding_box["row_min"]
    width  = bounding_box["col_max"] - bounding_box["col_min"]

    if width < 48 or height < 32:
        return True

    aspect_ratio = width / height

    if aspect_ratio > 1.5 or aspect_ratio < 0.7:
        return True

    return False


def save_cropped_segmented_output(model, device, loader, save_path, padding=True):
    with torch.no_grad():
        folder_type = save_path[:-1].split('/')[-1]
        segmentation_progress_bar = tqdm(loader, desc=f"Segmenting {folder_type}", unit=" batch")

        skipped = []

        output_results = []

        for data, _, file_names in segmentation_progress_bar:
            data = data.as_subclass(torch.Tensor)
            output = model(data.to(device)).argmax(dim=1).cpu()

            for image, segmentation, file_name in zip(data, output, file_names):
                image = image.squeeze(0).to(torch.float16)
                segmentation = segmentation.to(torch.uint8).clone()

                bounding_box = find_bounding_box(segmentation, 5)

                if remove_outlier_data(bounding_box):
                    skipped.append(file_name)
                    continue

                file_path = os.path.join(save_path, file_name.removesuffix(".npy") + ".pt")
                output_results.append((
                    apply_bounding_box(image, bounding_box, padding=padding),
                    apply_bounding_box(segmentation, bounding_box, padding=padding),
                    file_path
                ))

        save_progress_bar = tqdm(output_results, desc=f"Saving {folder_type}", unit=" slice")

        for image, mask, file_path in save_progress_bar:
            torch.save([image, mask], file_path) 

        if len(skipped) > 0:
            print(f"\nSkipped {len(skipped)} slices with invalid segmentation")

def load_postprocessed_data(pre_path="") -> list[tuple[torch.Tensor, torch.Tensor]]:
    data_dir = pre_path + "data/postprocessed/"

    slices_pairs = []
    total_entries = []

    for folder_name in ("training", "testing"):
        postprocessed_dir = os.path.join(data_dir, folder_name)
        
        entries = os.scandir(postprocessed_dir)
        sorted_entries = sorted(entries, key=lambda x: x.name.lower())
        total_entries += sorted_entries

        for entry in sorted_entries:
            slices_pairs.append(torch.load(entry))

    return slices_pairs, total_entries

if __name__ == "__main__":
    print(len(load_postprocessed_data()))