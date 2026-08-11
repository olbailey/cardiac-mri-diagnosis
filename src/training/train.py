import yaml

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import MriDataset, create_transforms, get_dataloaders

TRAIN_PATH = "data/raw/training"
TEST_PATH = "data/raw/testing"

BATCH_SIZE = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transform, test_transform = create_transforms()

train_dataset = MriDataset(TRAIN_PATH, train_transform)
test_dataset = MriDataset(TEST_PATH, test_transform)

train_loader, test_loader = get_dataloaders(train_dataset, test_dataset, BATCH_SIZE, device)
