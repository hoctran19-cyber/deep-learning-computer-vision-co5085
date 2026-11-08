"""Reproducibility, device, checkpoint, and result-serialization helpers."""

import csv
import json
import random
from pathlib import Path
from typing import Any

import torch
from torch import nn


def set_seed(seed: int) -> None:
    """Seed Python and PyTorch random number generators."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Prefer CUDA when available and otherwise use CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: nn.Module) -> int:
    """Count trainable scalar parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def resolve_path(path: str | Path, project_root: str | Path) -> Path:
    """Resolve relative configured paths against the project root."""
    path = Path(path)
    return path if path.is_absolute() else Path(project_root) / path


def save_checkpoint(checkpoint: dict[str, Any], path: str | Path) -> None:
    """Persist a checkpoint mapping, creating its parent directory as needed."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, destination)


def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    """Load checkpoint tensors onto the requested device."""
    return torch.load(path, map_location=device)


def save_history(history: dict[str, Any], path: str | Path) -> None:
    """Save epoch history as JSON or as one CSV row per epoch."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix.lower() == ".json":
        with destination.open("w", encoding="utf-8") as file:
            json.dump(history, file, indent=2)
        return

    fields = ("epoch", "train_loss", "train_accuracy", "validation_loss", "validation_accuracy")
    epoch_count = len(history["train_loss"])
    with destination.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for index in range(epoch_count):
            writer.writerow({
                field: (index + 1 if field == "epoch" else history[field][index])
                for field in fields
            })


def save_experiment_results(
    result: dict[str, Any], csv_path: str | Path, json_path: str | Path
) -> None:
    """Upsert one model's result into the E1-style comparison CSV and JSON."""
    model_name = str(result["model_name"])
    csv_destination, json_destination = Path(csv_path), Path(json_path)
    csv_destination.parent.mkdir(parents=True, exist_ok=True)
    json_destination.parent.mkdir(parents=True, exist_ok=True)

    rows: dict[str, dict[str, Any]] = {}
    if csv_destination.exists():
        with csv_destination.open(newline="", encoding="utf-8") as file:
            rows = {row["model_name"]: row for row in csv.DictReader(file)}
    fields = (
        "model_name",
        "test_loss",
        "test_accuracy",
        "num_parameters",
        "best_val_accuracy",
        "training_time",
    )
    rows[model_name] = {field: result[field] for field in fields}
    with csv_destination.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows.values())

    existing: dict[str, Any] = {}
    if json_destination.exists():
        with json_destination.open(encoding="utf-8") as file:
            existing = json.load(file)
    existing[model_name] = {field: result[field] for field in fields if field != "model_name"}
    with json_destination.open("w", encoding="utf-8") as file:
        json.dump(existing, file, indent=2)
