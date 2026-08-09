import os

import numpy as np
import SimpleITK as sitk

from utils.data import get_patient_frame, get_patient_label, resize_image

sitk.ProcessObject.SetGlobalWarningDisplay(False)


ROOT_DIR = "data/raw"

class patient():
    def __init__(self, image: sitk.Image, volume: np.ndarray, mask: np.ndarray, label: str):
        self.volume = volume
        self.mask = mask
        self.label = label
        self.voxel_size: tuple[float] = image.GetSpacing()
        self.resized_volume = resize_image(image, 1.5, is_mask=False)

    def get_num_slices(self):
        return self.volume.shape[0]

    def get_image_size(self):
        return self.volume.shape[1:]

    def get_voxel_sizes(self):
        return np.array(self.voxel_size[:2])

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

    image_volume = sitk.ReadImage(image_volume_path)

    image_volume_np = sitk.GetArrayFromImage(image_volume)   # (D, H, W)

    mask_volume = sitk.ReadImage(mask_volume_path)
    mask_volume = sitk.GetArrayFromImage(mask_volume)   # (D, H, W)

    return image_volume, image_volume_np, mask_volume, label
    

if __name__ == "__main__":
    patients = load_data()
    image_sizes = []
    for person in patients:
        image_sizes.append(person.resized_volume.GetSize()[:2])

    arr = np.array(image_sizes)

    print(f"max: {np.max(arr, axis=0), np.argmax(arr, axis=0)}")
    print(f"min: {np.min(arr, axis=0), np.argmin(arr, axis=0)}")
    print(f"median: {np.median(arr, axis=0)}")
    print(f"mean: {np.mean(arr, axis=0)}")
    print(f"std: {np.std(arr, axis=0)}")