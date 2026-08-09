import os
import shutil

import numpy as np
import nibabel as nib
import SimpleITK as sitk

from utils.data import get_patient_frame, get_patient_label, resize_image

sitk.ProcessObject.SetGlobalWarningDisplay(False)

DESTINATION = "data/processed"
ROOT_DIR = "data/raw"
VOXEL_SPACING_MEAN = 1.5

def get_nii_file(nii_file_path, is_mask) -> np.memmap:
    img = sitk.ReadImage(nii_file_path)
    processed_img = resize_image(img, VOXEL_SPACING_MEAN, is_mask=is_mask)
    img_data = sitk.GetArrayFromImage(processed_img) 
    return img_data

def process_patient(patient_path, patient_name, dest_images, dest_masks):
    disease = get_patient_label(patient_path)
    volume_4d_path = os.path.join(patient_path, patient_name + "_4d.nii")
    std, mean = calculate_patient_stats(volume_4d_path)

    first_frame = get_patient_frame(patient_path)
    image_volume_path = os.path.join(patient_path, patient_name + "_frame" + first_frame + ".nii")
    mask_volume_path = os.path.join(patient_path, patient_name + "_frame" + first_frame + "_gt.nii")

    image_volume = get_nii_file(image_volume_path, is_mask=False)
    image_volume = (image_volume - mean) / (std + 1e-8)
    mask_volume = get_nii_file(mask_volume_path, is_mask=True)

    slices_num = image_volume.shape[0]

    for i in range(slices_num):
        file_path_image = os.path.join(dest_images, f"{patient_name}_slice{i:02d}_{disease}.npy")
        file_path_mask = os.path.join(dest_masks, f"{patient_name}_slice{i:02d}_{disease}.npy")
        np.save(file_path_image, image_volume[i, :, :])
        np.save(file_path_mask, mask_volume[i, :, :])

def calculate_patient_stats(file_4d_path):
    img = nib.load(file_4d_path)
    volume_4d = img.get_fdata()
    return np.std(volume_4d), np.mean(volume_4d)

def process_folder(folder_name):
    raw_dir = os.path.join(ROOT_DIR, folder_name)
    processed_dir = os.path.join(DESTINATION, folder_name)
    processed_images = os.path.join(processed_dir, "images")
    processed_masks = os.path.join(processed_dir, "masks")
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(processed_images, exist_ok=True)
    os.makedirs(processed_masks, exist_ok=True)

    entries = os.scandir(raw_dir)
    sorted_entries = sorted(entries, key=lambda x: x.name.lower())

    for entry in sorted_entries:
        if entry.is_dir(follow_symlinks=False):
            process_patient(entry.path, entry.name, processed_images, processed_masks)


def main():
    if "processed" in os.listdir("data"):
        shutil.rmtree(DESTINATION)
    os.makedirs(DESTINATION, exist_ok=True)
    print("This will take a minute...")
    process_folder("training")
    print("Processed training data.")
    process_folder("testing")
    print("Processed testing data.")

if __name__ == "__main__":
    main()