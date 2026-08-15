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


def create_transforms(image_size: int) -> A.Compose:
    """_summary_

    Args:
        image_size (int): Set image size for the input for the model,
        will output a square image, image size must be a multiple of 2^k, where k is the number of U-net pooling steps

    Returns:
        A.Compose: _description_
    """
    basic_transform = [
        A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=0, value=0),
        A.CenterCrop(height=image_size, width=image_size),
        ToTensorV2(),
    ]

    train_transform = A.Compose(
        [
            
        ]
        + basic_transform
    )

    test_transform = A.Compose(basic_transform)

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