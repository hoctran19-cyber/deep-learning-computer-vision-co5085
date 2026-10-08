"""Small reusable utilities for reproducibility, persistence, and plots."""

import json
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch import nn


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Select CUDA when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def project_path(path: str | Path, project_root: Path) -> Path:
    """Resolve a project-relative path without relying on a machine-specific cwd."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else project_root / candidate


def count_trainable_parameters(model: nn.Module) -> int:
    """Count parameters with gradients enabled."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def save_checkpoint(model: nn.Module, path: str | Path, optimizer: torch.optim.Optimizer | None = None,
                    epoch: int | None = None) -> None:
    """Save model state and optional optimizer metadata."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {"model_state_dict": model.state_dict()}
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if epoch is not None:
        checkpoint["epoch"] = epoch
    torch.save(checkpoint, path)


def load_checkpoint(model: nn.Module, path: str | Path, device: torch.device) -> dict[str, Any]:
    """Load a checkpoint into a model and return its metadata."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def save_metrics(metrics: dict[str, Any], path: str | Path) -> None:
    """Save JSON-serializable metrics, excluding tensors and arrays if needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, default=lambda value: value.tolist())


def plot_training_curves(history: dict[str, list[float]], path: str | Path) -> None:
    """Plot loss and accuracy curves from a training history."""
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["validation_loss"], label="Validation")
    axes[0].set_title("Loss")
    axes[1].plot(history["train_accuracy"], label="Train")
    axes[1].plot(history["validation_accuracy"], label="Validation")
    axes[1].set_title("Accuracy")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.legend()
    figure.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)


def plot_confusion_matrix(matrix: np.ndarray, class_names: list[str], path: str | Path) -> None:
    """Plot a confusion matrix without creating any metrics itself."""
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(xticks=range(len(class_names)), yticks=range(len(class_names)),
             xticklabels=class_names, yticklabels=class_names,
             xlabel="Predicted label", ylabel="True label")
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right")
    figure.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)


def plot_misclassified_samples(samples: list[tuple[torch.Tensor, int, int]],
                               class_names: list[str], path: str | Path,
                               max_samples: int = 16) -> None:
    """Plot a selection of misclassified images."""
    selected = samples[:max_samples]
    if not selected:
        return
    figure, axes = plt.subplots(4, 4, figsize=(8, 8))
    for axis, (image, true_label, predicted_label) in zip(axes.flat, selected):
        display_image = image.squeeze().numpy()
        if display_image.ndim == 3:
            display_image = np.transpose(display_image, (1, 2, 0))
        axis.imshow(display_image, cmap="gray" if display_image.ndim == 2 else None)
        axis.set_title(f"T: {class_names[true_label]}\nP: {class_names[predicted_label]}", fontsize=8)
        axis.axis("off")
    for axis in axes.flat[len(selected):]:
        axis.axis("off")
    figure.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)
