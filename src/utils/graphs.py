import torch
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt

def draw_graph(df: pd.DataFrame, data_label: str, title: str, y_label: str, x_label, length=20, colour="#1f77b4", other_values=[]) -> None:
    plt.figure(figsize=(length, 5))

    if other_values is not None:
        plt.plot(df["Datetime"], df[data_label], label=data_label)
        for item_label in other_values:
            plt.plot(df["Datetime"], df[item_label], label=item_label)
    else:
        plt.plot(df["Datetime"], df[data_label], label=data_label, color=colour)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_predictions(model, loader: DataLoader, device: torch.device, num_points: int = None, title: str = "Predicted vs Actual"):
    """
    Runs the model over a DataLoader and plots predicted vs actual values.

    num_points: if set, only plots the first N points (useful for zooming in
                on long time series where plotting everything is unreadable).
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs).squeeze(-1)

            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())

    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()

    if num_points is not None and num_points > 0:
        preds = preds[:num_points]
        targets = targets[:num_points]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Top plot: predicted vs actual over "time" (i.e. sample index)
    axes[0].plot(targets, label='Actual', linewidth=1.5, alpha=0.8)
    axes[0].plot(preds, label='Predicted', linewidth=1.5, alpha=0.8)
    axes[0].set_xlabel('Sample index')
    axes[0].set_ylabel('Value')
    axes[0].set_title(title)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Bottom plot: scatter of predicted vs actual (perfect predictions fall on y=x line)
    axes[1].scatter(targets, preds, alpha=0.4, s=10)
    min_val = min(targets.min(), preds.min())
    max_val = max(targets.max(), preds.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1, label='Perfect prediction')
    axes[1].set_xlabel('Actual')
    axes[1].set_ylabel('Predicted')
    axes[1].set_title('Predicted vs Actual (scatter)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    return preds, targets