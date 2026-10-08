"""Evaluation helpers for test metrics and model comparison."""

from typing import Any

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader

from .utils import count_trainable_parameters


def collect_predictions(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, list[torch.Tensor]]:
    """Collect labels, predictions, and CPU images from a loader."""
    model.eval()
    labels, predictions, images = [], [], []
    with torch.no_grad():
        for batch_images, targets in loader:
            output = model(batch_images.to(device))
            labels.extend(targets.numpy())
            predictions.extend(output.argmax(dim=1).cpu().numpy())
            images.extend(batch_images.cpu())
    return np.array(labels), np.array(predictions), images


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    """Return test accuracy and predictions without changing model weights."""
    labels, predictions, images = collect_predictions(model, loader, device)
    return {
        "accuracy": float((labels == predictions).mean()),
        "labels": labels,
        "predictions": predictions,
        "images": images,
        "confusion_matrix": confusion_matrix(labels, predictions),
        "trainable_parameters": count_trainable_parameters(model),
    }


def misclassified_samples(evaluation: dict[str, Any]) -> list[tuple[torch.Tensor, int, int]]:
    """Return image, true label, predicted label for incorrect predictions."""
    labels, predictions = evaluation["labels"], evaluation["predictions"]
    return [(image, int(label), int(prediction)) for image, label, prediction in zip(
        evaluation["images"], labels, predictions) if label != prediction
    ]


def compare_results(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Create a compact, sortable summary for multiple evaluated models."""
    return [
        {"model": name, "accuracy": values["accuracy"],
         "trainable_parameters": values["trainable_parameters"]}
        for name, values in results.items()
    ]
