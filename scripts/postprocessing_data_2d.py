import yaml
import os
import shutil

import torch
import segmentation_models_pytorch as smp

from dataset import SegmentationDataset, create_transforms, get_dataloaders
from utils.postprocessing import save_cropped_segmented_output

MODEL_NAME = "segmentation_default_model_save"
DEFAULT_MODEL_PATH = "output/segmentation/" + MODEL_NAME + "/checkpoint.pt"
POSTPROCESSING_DIR = "data/postprocessed"

TRAIN_FILE_PATH = "data/processed/training"
TEST_FILE_PATH = "data/processed/testing"

PADDING = True

def main():
    if "postprocessed" in os.listdir("data"):
        print("Deleting previously generated data...")
        shutil.rmtree(POSTPROCESSING_DIR)


    with open("configs/unet.yaml", 'r') as file:
        configs = yaml.safe_load(file)

    IMAGE_SIZE = configs["model"]["image_size"]
    BATCH_SIZE = configs["training"]["batch_size"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform, test_transform = create_transforms(IMAGE_SIZE)

    train_dataset = SegmentationDataset(TRAIN_FILE_PATH, train_transform)
    test_dataset = SegmentationDataset(TEST_FILE_PATH, test_transform)

    train_loader, test_loader = get_dataloaders(train_dataset, test_dataset, BATCH_SIZE, device)

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,   # no need to download ImageNet weights, they're overwritten below
        encoder_depth=5,
        in_channels=1,
        classes=4,
        activation=None,
    )

    state_dict = torch.load(DEFAULT_MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print()
    training_path = os.path.join(POSTPROCESSING_DIR, "training/")
    os.makedirs(os.path.dirname(training_path), exist_ok=True)

    save_cropped_segmented_output(model, device, train_loader, training_path, padding=PADDING)

    print()
    test_path = os.path.join(POSTPROCESSING_DIR, "testing/")
    os.makedirs(os.path.dirname(test_path), exist_ok=True)

    save_cropped_segmented_output(model, device, test_loader, test_path, padding=PADDING)

    print("\nPostprocessing complete.\n")


if __name__ == "__main__":
    main()