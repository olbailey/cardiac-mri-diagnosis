import os
import math

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, Compose
from monai.data import decollate_batch

from tqdm.auto import tqdm

# from .graphs import plot_predictions


def train_epoch(model, train_loader: DataLoader, loss_function, optimizer: optim.Adam, device: torch.device, print_interval_num=10):
    model.train()
    running_loss = 0
    num_batches = len(train_loader)
    print_interval = max(1, num_batches // print_interval_num)

    train_progress_bar = tqdm(train_loader, desc=f"Training", unit="batch")

    # for batch_idx, (data, target) in enumerate(train_loader):
    batch_idx = 0
    for data, target in train_progress_bar:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        predicted = model(data).squeeze(-1)

        loss = loss_function(predicted, target)
        loss.backward()
        optimizer.step()

        # Track progress
        running_loss += loss.item()

        if batch_idx % print_interval == 0 or (batch_idx + 1) == num_batches:
            avg_loss = running_loss / (batch_idx + 1)
            
            train_progress_bar.set_postfix(loss=f"{avg_loss:.3f}")
        batch_idx += 1

def evaluate(model: nn.Module, val_loader: DataLoader, classes_num: 4, device: torch.device):
    model.eval()

    # note: include_background=False is standard for reporting -
    # background Dice is usually near 1.0 and inflates your average meaninglessly
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

    # post-processing to convert model outputs -> discrete one-hot predictions
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=classes_num)])
    post_label = Compose([AsDiscrete(to_onehot=classes_num)])

    with torch.no_grad():
        val_progress_bar = tqdm(val_loader, desc=f"Validation", unit="batch")

        for inputs, targets in val_progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            predicted = model(inputs)

            outputs = decollate_batch(predicted)
            labels  = decollate_batch(targets)

            # apply post-processing: argmax -> one-hot for predictions, one-hot for labels
            outputs = [post_pred(o) for o in outputs]
            labels  = [post_label(l) for l in labels]

            # accumulate - this doesn't return a value yet, just adds to internal buffer
            dice_metric(y_pred=outputs, y=labels)

    # aggregate over the whole validation set
    mean_dice = dice_metric.aggregate().item()
    dice_metric.reset()  # reset for next epoch


    return mean_dice
    
class EarlyStopping:
    def __init__(self, temp_model_dir, patience, min_delta, restore_best_weights=True):
        self.current_best = math.inf
        self.temp_model_dir = temp_model_dir
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights

        self.count = 0
        self.stopped = False

    def update(self, model, value) -> nn.Module:
        if value < self.current_best and self.current_best - value > self.min_delta:
            self.save_model(model)
            print(f"New best, saving model...")
            self.current_best = value
            self.count = 0
        elif self.count < self.patience:
            self.count += 1 
            print(f"Model has regressed, current count: {self.count}/{self.patience}, Value Delta: {self.current_best - value:.8f}")
        elif self.restore_best_weights:
            model = self.restore_best_model(model)
            self.stopped = True

        return model
        
    def save_model(self, model):
        os.makedirs(self.temp_model_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(self.temp_model_dir, "model_temp_save.pt"))

    def restore_best_model(self, model:nn.Module):
        try:
            model.load_state_dict(torch.load(os.path.join(self.temp_model_dir, "model_temp_save.pt")))
        except FileNotFoundError:
            print("ERROR! Could not find model parameter file!")

        return model
    
    
def finish_training(data_dir: str, model: nn.Module):
    print()
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    answer = input("\nwould you like to save the model data? (y/n): ").strip().lower()
    if answer not in ("y", "yes"):
        return

    model_name = input("enter the file name for the model trained: ")
    torch.save(model.state_dict(), os.path.join(data_dir, model_name + ".pt"))

# def show_graph(model, val_loader, device, overide_show=False):
#     if not overide_show:
#         try:
#             x = int(input("How many points would you like to plot? "))
#             plot_predictions(model, val_loader, device, num_points=x)
#         except ValueError:
#             pass
