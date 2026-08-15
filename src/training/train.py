import yaml

import torch
import torch.nn as nn
import torch.optim as optim

from monai.networks.nets import UNet as monia_Unet
from monai.losses import DiceCELoss

import segmentation_models_pytorch as smp

from dataset import MriDataset, create_transforms, get_dataloaders
from model import UNet

TRAIN_PATH = "data/raw/training"
TEST_PATH = "data/raw/testing"

BATCH_SIZE = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transform, test_transform = create_transforms(256)

train_dataset = MriDataset(TRAIN_PATH, train_transform)
test_dataset = MriDataset(TEST_PATH, test_transform)

train_loader, test_loader = get_dataloaders(train_dataset, test_dataset, BATCH_SIZE, device)

model = UNet(num_classes=4, encoder_depth=4, base_c=64)

# Monia U-Net models and loss
model = monia_Unet(
    spatial_dims=2,
    in_channels=1,
    out_channels=4,
    channels=(32, 64, 128, 256, 512),
    strides=(2, 2, 2, 2),
    num_res_units=2
)
loss_fn = DiceCELoss(sigmoid=True)

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    encoder_depth=4,
    in_channels=1,
    classes=4,
    activation=None
)
