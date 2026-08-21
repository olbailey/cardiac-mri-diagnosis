import os

import numpy as np
import nibabel as nib

import torch
from torch.utils.data import Dataset, Subset, DataLoader
import monai
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped, ResizeWithPadOrCropd, NormalizeIntensityd,
    RandAffined, RandFlipd, RandGaussianNoised, RandAdjustContrastd
)

class MriDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.path_images = os.path.join(root_dir, "images")
        self.path_masks = os.path.join(root_dir, "masks")

        self.file_names = os.listdir(self.path_images)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]

        data_dict = {
            "image": os.path.join(self.path_images, file_name), 
            "label": os.path.join(self.path_masks, file_name)
        }
        if self.transform is not None:
            data_dict = self.transform(data_dict)

        image_tensor, mask_tensor = data_dict["image"], data_dict["label"]

        # print(type(image_tensor))
        # raise Exception()

        return image_tensor, mask_tensor

    def retreive_file(self, path, file_name):
        path = os.path.join(path, file_name)
        return np.load(path)


def create_transforms(image_size: int) -> monai.transforms.Compose:
    """_summary_

    Args:
        image_size (int): Set image size for the input for the model,
        will output a square image, image size must be a multiple of 2^k, where k is the number of U-net pooling steps

    Returns:
        monai.transforms.Compose: _description_
    """
    basic_transform = [
        LoadImaged(keys=["image", "label"]), # Loading py file
        EnsureChannelFirstd(keys=["image", "label"]), # Adding dimension to be expected format (c, h, w)
        ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=(image_size, image_size), mode="constant"), # Padding Or cropping to expected size
    ]

    train_transform = Compose(
        basic_transform +
        [
            # RandAffined(keys=["image", "label"], prob=0.3, rotate_range=(0.1, 0.1), scale_range=(0.1, 0.1), mode=("bilinear", "nearest")), # Rotating and Scaling the image
            # RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0), # 
            # RandGaussianNoised(keys=["image"], prob=0.2, std=0.01), # Gaussian noise
            # RandAdjustContrastd(keys=["image"], prob=0.2, gamma=(0.8, 1.2)),
            EnsureTyped(keys=["image", "label"], dtype=[torch.float32, torch.long])
        ]
    )

    test_transform = Compose(
        basic_transform + [EnsureTyped(keys=["image", "label"], dtype=[torch.float32, torch.long])]
    )

    return train_transform, test_transform

def get_dataloaders(train_dataset, test_dataset, batch_size, device):
    if device.type == "cuda":
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2)
    else:
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


if __name__ == "__main__":
    dataset = MriDataset("data/processed/training")
    # dataset[0]
    for i in range(len(dataset)):
        dataset[i]