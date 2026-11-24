import os

import numpy as np
import torch

from tqdm.auto import tqdm


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

def apply_bounding_box(tensor: torch.Tensor, box):
    return tensor[box["row_min"]:box["row_max"], box["col_min"]:box["col_max"]].clone()

def save_cropped_segmented_output(model, device, loader, save_path):
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

                if bounding_box is None:
                    skipped.append(file_name)
                    continue

                file_path = os.path.join(save_path, file_name.removesuffix(".npy") + ".pt")
                output_results.append((
                    apply_bounding_box(image, bounding_box),
                    apply_bounding_box(segmentation, bounding_box),
                    file_path
                ))

        save_progress_bar = tqdm(output_results, desc=f"Saving {folder_type}", unit=" slice")

        for image, mask, file_path in save_progress_bar:
            torch.save([image, mask], file_path) 

        if len(skipped) > 0:
            print(f"\nSkipped {len(skipped)} slices with empty segmentation, e.g. {skipped[:3]}")