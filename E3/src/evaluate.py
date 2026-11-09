"""Test-set evaluation of the best validation-selected E3 checkpoint."""

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from .utils import count_parameters, get_device, load_checkpoint


def evaluate_model(
    model: nn.Module, loader: DataLoader, device: torch.device | None = None
) -> dict[str, float | int]:
    """Calculate sample-weighted cross-entropy, accuracy, and parameter count."""
    if len(loader) == 0:
        raise ValueError("Cannot evaluate an empty data loader.")
    device = device or get_device()
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = total_correct = total_items = 0
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            item_count = targets.size(0)
            total_loss += criterion(logits, targets).item() * item_count
            total_correct += int((logits.argmax(dim=1) == targets).sum().item())
            total_items += item_count
    return {
        "test_loss": total_loss / total_items,
        "test_accuracy": total_correct / total_items,
        "num_parameters": count_parameters(model),
    }


def evaluate_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    test_loader: DataLoader,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Load and evaluate a best checkpoint, returning its experiment metadata."""
    device = device or get_device()
    checkpoint = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = evaluate_model(model, test_loader, device)
    return {
        "model_name": checkpoint["model_name"],
        **metrics,
        "best_val_accuracy": float(checkpoint["best_val_accuracy"]),
        "training_time": float(checkpoint["training_time"]),
    }
