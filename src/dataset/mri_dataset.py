import os

import numpy as np
import nibabel as nib

import torch
from torch.utils.data import Dataset, Subset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

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

        image_np = self.retreive_file(self.path_images, file_name)
        image_tensor = torch.from_numpy(image_np).float()
        image_tensor = image_tensor.unsqueeze(0)

        if self.transform is not None:
            image_tensor = self.transform(image_tensor)

        mask_np = self.retreive_file(self.path_masks, file_name)
        mask_tensor = torch.from_numpy(mask_np).long()

        return image_tensor, mask_tensor

    def retreive_file(self, path, file_name):
        path = os.path.join(path, file_name)
        return np.load(path)


def create_transforms():
    train_transform = A.Compose([
        A.LongestMaxSize(max_size=500),          # cap the largest side
        A.PadIfNeeded(min_height=500, min_width=500, border_mode=0, value=0),
        A.CenterCrop(height=384, width=384),      # or whatever your model input is
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        ToTensorV2(),
    ])

    test_transform = A.Compose([
        A.LongestMaxSize(max_size=500),
        A.PadIfNeeded(min_height=500, min_width=500, border_mode=0, value=0),
        A.CenterCrop(height=384, width=384),
        ToTensorV2(),
    ])

    return train_transform, test_transform


if __name__ == "__main__":
    dataset = MriDataset("data/processed/training")
    # dataset[0]
    for i in range(len(dataset)):
        dataset[i]