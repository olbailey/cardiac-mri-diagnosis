import yaml

import torch
import torch.nn as nn
import torch.optim as optim

from monai.networks.nets import UNet as monia_Unet
from monai.losses import DiceCELoss

import segmentation_models_pytorch as smp

from dataset import SegmentationDataset, create_transforms, get_dataloaders
from utils.training import train_epoch, evaluate, finish_training, EarlyStopping
from utils.graphs import show_single_slice_prediction
from utils.segmentation.args_parser import parse_args

def main():

    MODEL_DIR = "output/segmentation"

    args, configs = parse_args(base_dir=MODEL_DIR)

    MODEL_DATA_NAME = args.model_name
    TRAIN_FILE_PATH = args.train_path
    TEST_FILE_PATH = args.test_path

    CLASSES_NUM = configs["model"]["classes"]
    IMAGE_SIZE = configs["model"]["image_size"]
    ENCODER_DEPTH = configs["model"]["encoder_depth"]
    BASE_CHANNELS = configs["model"]["base_channels"]

    LEARNING_RATE = args.lr
    BATCH_SIZE = args.batch_size

    ENABLE_MODEL_SAVING = args.save_model
    TESTING_SLICE_NAME = args.test_slice
    NUM_EPOCHS = args.epochs


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform, test_transform = create_transforms(IMAGE_SIZE)

    train_dataset = SegmentationDataset(TRAIN_FILE_PATH, train_transform)
    test_dataset = SegmentationDataset(TEST_FILE_PATH, test_transform)

    train_loader, test_loader = get_dataloaders(train_dataset, test_dataset, BATCH_SIZE, device)

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        encoder_depth=5,
        in_channels=1,
        classes=4,
        activation=None,
    ).to(device)

    # Monia U-Net models and loss
    # model = monia_Unet(
    #     spatial_dims=2,
    #     in_channels=1,
    #     out_channels=CLASSES_NUM,
    #     channels=(32, 64, 128, 256, 512),
    #     strides=(2, 2, 2, 2),
    #     num_res_units=2
    # ).to(device)

    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)

    optimizer = optim.Adam(model.parameters(), LEARNING_RATE, weight_decay=1e-5)
    # schedular_steplr = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    schedular_plateau = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode="min", 
        threshold_mode="abs", 
        threshold=configs["training"]["min_delta"], 
        factor=0.2, 
        patience=2
    )

    early_stopping = EarlyStopping(
        MODEL_DATA_NAME, 
        patience=configs["training"]["patience"], 
        min_delta=configs["training"]["min_delta"]
    )


    try:
        for epoch in range(NUM_EPOCHS):
            print(f"\nEpoch: {epoch + 1}")
            train_loss = train_epoch(model, train_loader, loss_function, optimizer, device)

            mean_dice, dice_classes_str, eval_loss = evaluate(model, test_loader, loss_function, CLASSES_NUM, device)
            print(f"Training Loss: {train_loss:.4f}, Validation Loss: {eval_loss:.4f}")
            print(f"Validation Dice Metric {dice_classes_str}") #:.6f
            print(f"lr: {schedular_plateau.get_last_lr()[0]:.6f}")

            model = early_stopping.update(model, mean_dice)

            # schedular_steplr.step()

            schedular_plateau.step(mean_dice)


            if early_stopping.stopped:
                # show_graph(model, test_loader, device)
                # finish_training(MODEL_DIR, model)
                break
            
    except KeyboardInterrupt:
        show_single_slice_prediction(model, test_dataset, TESTING_SLICE_NAME, CLASSES_NUM, device)

        if ENABLE_MODEL_SAVING:
            finish_training(MODEL_DIR, model)

if __name__ == "__main__":
    main()