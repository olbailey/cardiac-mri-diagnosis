import yaml

import torch
import torch.nn as nn
import torch.optim as optim

from monai.networks.nets import UNet as monia_Unet
from monai.losses import DiceCELoss

import segmentation_models_pytorch as smp

from dataset import MriDataset, create_transforms, get_dataloaders
from model import UNet
from utils.training import train_epoch, evaluate, finish_training, EarlyStopping
from utils.graphs import show_single_slice_prediction

with open("configs/unet.yaml", 'r') as file:
    configs = yaml.safe_load(file)

MODEL_DATA_DIR = configs["data"]["model_dir"]
MODEL_TEMP_DATA_DIR = "outputs/models/temp"
TRAIN_FILE_PATH = "data/processed/training"
TEST_FILE_PATH = "data/processed/testing"

CLASSES_NUM = configs["model"]["classes"]
IMAGE_SIZE = configs["model"]["image_size"]
ENCODER_DEPTH = configs["model"]["encoder_depth"]
BASE_CHANNELS = configs["model"]["base_channels"]

LEARNING_RATE = configs["training"]["learning_rate"]
BATCH_SIZE = configs["training"]["batch_size"]

ENABLE_MODEL_SAVING = False
OVERIDE_SHOWING_GRAPHS = True
MODEL_COMPUTE_VALUE_DELTAS = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transform, test_transform = create_transforms(IMAGE_SIZE)

train_dataset = MriDataset(TRAIN_FILE_PATH, train_transform)
test_dataset = MriDataset(TEST_FILE_PATH, test_transform)

train_loader, test_loader = get_dataloaders(train_dataset, test_dataset, BATCH_SIZE, device)

image, label = next(iter(train_loader))

# model = UNet(num_classes=CLASSES_NUM, encoder_depth=ENCODER_DEPTH, base_c=64)

# model = smp.Unet(
#     encoder_name="resnet34",
#     encoder_weights="imagenet",
#     encoder_depth=4,
#     in_channels=1,
#     classes=4,
#     activation=None
# )

# Monia U-Net models and loss
model = monia_Unet(
    spatial_dims=2,
    in_channels=1,
    out_channels=CLASSES_NUM,
    channels=(32, 64, 128, 256, 512),
    strides=(2, 2, 2, 2),
    num_res_units=2
).to(device)
loss_function = DiceCELoss(to_onehot_y=True, softmax=True)

optimizer = optim.Adam(model.parameters(), LEARNING_RATE, weight_decay=1e-5)
schedular_steplr = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
schedular_plateau = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode="min", 
    threshold_mode="abs", 
    threshold=configs["training"]["min_delta"], 
    factor=0.2, 
    patience=2
)

early_stopping = EarlyStopping(
    MODEL_TEMP_DATA_DIR, 
    patience=configs["training"]["patience"], 
    min_delta=configs["training"]["min_delta"]
)

num_epochs = 50

try:
    for epoch in range(num_epochs):
        print(f"\nEpoch: {epoch + 1}")
        train_loss = train_epoch(model, train_loader, loss_function, optimizer, device)

        dice, eval_loss = evaluate(model, test_loader, loss_function, CLASSES_NUM, device)
        print(f"Training Loss: {train_loss:.4f}, Validation Loss: {eval_loss:.4f}")
        print(f"Validation Dice Metric {{{dice}}}, lr: {schedular_plateau.get_last_lr()[0]:.6f}") #:.6f

        # model = early_stopping.update(model, dice)

        # schedular_plateau.step(dice)

        # show_graph(model, test_loader, device, overide_show=OVERIDE_SHOWING_GRAPHS)

        # if early_stopping.stopped:
        #     # show_graph(model, test_loader, device)
        #     finish_training(MODEL_DATA_DIR, model)
        #     break
        
except KeyboardInterrupt:
    if ENABLE_MODEL_SAVING:
        finish_training(MODEL_DATA_DIR, model)