import os

import numpy as np
import nibabel as nib

DESTINATION = "data/processed"
ROOT_DIR = "data/raw"


def get_nii_file(nii_file_path):
    img = nib.load(nii_file_path)
    return img.get_fdata()

def get_label(patient_dir_path):
    config_path = os.path.join(patient_dir_path, "Info.cfg")
    with open(config_path, 'r') as config_file:
        for line in config_file.readlines():
            if "Group" in line:
                # line including the group will be structured as "Group: type"
                return line.strip().split(": ")[1]

        raise RuntimeError("Error! 'Group' config not found")
        

def get_frame(patient_path, finding_lowest=True):
    frame_digits = []
    for file_name in os.listdir(patient_path):
        if "frame" in file_name:
            # frame file with be structured for example patient001_frame01...
            # so split to get 01... then retrieve the first two characters to get the number
            frame_digit = file_name.split("frame")[1][:2]
            frame_digits.append(frame_digit)

    if finding_lowest:
        lowest_frame_digit = min(list(map(lambda x: int(x), frame_digits))) 
    else:
        lowest_frame_digit = max(list(map(lambda x: int(x), frame_digits))) 

    
    for digit in frame_digits:
        if f"{lowest_frame_digit:02d}" == digit:
            return digit

    raise RuntimeError("Error! Lowest frame not found")

def process_patient(patient_path, patient_name, dest_images, dest_masks):
    disease = get_label(patient_path)

    first_frame = get_frame(patient_path)
    image_volume_path = os.path.join(patient_path, patient_name + "_frame" + first_frame + ".nii")
    mask_volume_path = os.path.join(patient_path, patient_name + "_frame" + first_frame + "_gt.nii")

    image_volume = get_nii_file(image_volume_path)
    mask_volume = get_nii_file(mask_volume_path)

    slices_num = image_volume.shape[2]

    for i in range(slices_num):
        file_path_image = os.path.join(dest_images, f"{patient_name}_slice{i:02d}_{disease}.npy")
        file_path_mask = os.path.join(dest_masks, f"{patient_name}_slice{i:02d}_{disease}_gt.npy")
        np.save(file_path_image, image_volume[:, :, i])
        np.save(file_path_mask, mask_volume[:, :, i])

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
    os.makedirs(DESTINATION, exist_ok=True)

    process_folder("training")
    process_folder("testing")

if __name__ == "__main__":
    main()