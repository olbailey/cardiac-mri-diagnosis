import os

import numpy as np
import nibabel as nib

from utils.data import get_patient_frame, get_patient_label

ROOT_DIR = "data/raw"

class patient():
    def __init__(self, volume, mask, label):
        self.volume = volume
        self.mask = mask
        self.label = label

    def get_num_slices(self):
        return self.volume.shape[2]

    def get_image_size(self):
        return self.volume.shape[:2]

def load_data() -> list[patient]:
    patients = []

    for folder_name in ("training", "testing"):
        raw_dir = os.path.join(ROOT_DIR, folder_name)
        
        entries = os.scandir(raw_dir)
        sorted_entries = sorted(entries, key=lambda x: x.name.lower())

        for entry in sorted_entries:
            if entry.is_dir(follow_symlinks=False):
                patients.append(patient(*process_patient(entry.path, entry.name)))

    return patients
        

def process_patient(patient_path, patient_name):
    label = get_patient_label(patient_path)
    first_frame = get_patient_frame(patient_path)
    image_volume_path = os.path.join(patient_path, patient_name + "_frame" + first_frame + ".nii")
    mask_volume_path = os.path.join(patient_path, patient_name + "_frame" + first_frame + "_gt.nii")

    image_volume = nib.load(image_volume_path).get_fdata()
    mask_volume = nib.load(mask_volume_path).get_fdata()

    return image_volume, mask_volume, label
    

if __name__ == "__main__":
    patients = load_data()
    image_sizes = []
    for person in patients:
        image_sizes.append(person.get_image_size())

    arr = np.array(image_sizes)

    print(np.max(arr, axis=0), np.argmax(arr, axis=0))
    print(np.min(arr, axis=0), np.argmin(arr, axis=0))