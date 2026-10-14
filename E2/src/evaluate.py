"""Đánh giá mô hình E2."""

from __future__ import annotations

import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from .utils import count_trainable_parameters


def accuracy_count(logits: torch.Tensor, targets: torch.Tensor) -> int:
    return int((logits.argmax(dim=1) == targets).sum().item())


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device,
                   num_classes: int) -> dict[str, object]:
    """Tính test accuracy, loss, confusion matrix và thời gian suy luận."""
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = total_correct = total_items = 0
    matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    start = time.perf_counter()
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            loss = criterion(logits, targets)
            total_loss += loss.item() * targets.size(0)
            total_correct += accuracy_count(logits, targets)
            total_items += targets.size(0)
            predictions = logits.argmax(dim=1).cpu()
            for true_label, predicted_label in zip(targets.cpu(), predictions):
                matrix[int(true_label), int(predicted_label)] += 1
    inference_time = time.perf_counter() - start
    return {
        "loss": total_loss / total_items,
        "accuracy": total_correct / total_items,
        "confusion_matrix": matrix.tolist(),
        "inference_time": inference_time,
        "trainable_parameters": count_trainable_parameters(model),
    }
