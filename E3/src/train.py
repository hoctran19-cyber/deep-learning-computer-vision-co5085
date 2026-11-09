"""Reusable training loop for E3 LSTM and GRU classifiers."""

import copy
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .utils import get_device, save_checkpoint, set_seed


def _epoch_metrics(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optimizer | None,
) -> tuple[float, float]:
    if len(loader) == 0:
        raise ValueError("Cannot process an empty data loader.")
    training = optimizer is not None
    model.train(training)
    total_loss = total_correct = total_items = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            if optimizer is not None:
                optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            item_count = targets.size(0)
            total_loss += loss.item() * item_count
            total_correct += int((logits.argmax(dim=1) == targets).sum().item())
            total_items += item_count
    return total_loss / total_items, total_correct / total_items


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch and return sample-weighted loss and accuracy."""
    return _epoch_metrics(model, loader, criterion, device, optimizer)


def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Validate without computing gradients or updating model parameters."""
    return _epoch_metrics(model, loader, criterion, device, None)


def train_model(
    model: nn.Module,
    model_name: str,
    loaders: dict[str, DataLoader],
    config: dict[str, Any],
    checkpoint_path: str | Path,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Train using validation accuracy for checkpoint selection; never access test."""
    model_name = model_name.lower()
    if model_name not in {"lstm", "gru"}:
        raise ValueError("model_name must be 'lstm' or 'gru'.")
    if "train" not in loaders or "validation" not in loaders:
        raise KeyError("loaders must contain 'train' and 'validation'.")

    seed = int(config.get("seed", 42))
    set_seed(seed)
    device = device or get_device()
    model.to(device)
    training_config = config.get("training", {})
    if str(training_config.get("optimizer", "adam")).lower() != "adam":
        raise ValueError("E3 supports the Adam optimizer.")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_config.get("learning_rate", 1e-3)),
        weight_decay=float(training_config.get("weight_decay", 0.0)),
    )
    criterion = nn.CrossEntropyLoss()
    history: dict[str, Any] = {
        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": [],
    }
    best_val_accuracy = -math.inf
    best_epoch = 0
    best_model_state: dict[str, torch.Tensor] | None = None
    best_optimizer_state: dict[str, Any] | None = None

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start_time = time.perf_counter()
    epochs = int(training_config.get("epochs", 10))
    if epochs <= 0:
        raise ValueError("training.epochs must be positive.")
    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device
        )
        validation_loss, validation_accuracy = validate_one_epoch(
            model, loaders["validation"], criterion, device
        )
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["validation_loss"].append(validation_loss)
        history["validation_accuracy"].append(validation_accuracy)
        if validation_accuracy > best_val_accuracy:
            best_val_accuracy = validation_accuracy
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())
            save_checkpoint({
                "model_state_dict": best_model_state,
                "optimizer_state_dict": best_optimizer_state,
                "model_name": model_name,
                "epoch": epoch,
                "best_val_accuracy": best_val_accuracy,
                "config": copy.deepcopy(config),
                "seed": seed,
            }, checkpoint_path)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_time = time.perf_counter() - start_time
    if best_model_state is None or best_optimizer_state is None:
        raise RuntimeError("Training did not produce a valid best checkpoint.")
    checkpoint = {
        "model_state_dict": best_model_state,
        "optimizer_state_dict": best_optimizer_state,
        "model_name": model_name,
        "epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "training_time": training_time,
        "config": copy.deepcopy(config),
        "seed": seed,
    }
    save_checkpoint(checkpoint, checkpoint_path)
    history["best_epoch"] = best_epoch
    history["best_val_accuracy"] = best_val_accuracy
    history["training_time"] = training_time
    return history
