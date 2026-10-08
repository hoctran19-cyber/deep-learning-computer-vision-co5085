"""Manual PyTorch training loop and command-line experiment entry point."""

import argparse
import csv
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .datasets import create_dataloaders, load_datasets
from .evaluate import evaluate_model, misclassified_samples
from .models import get_model
from .utils import (count_trainable_parameters, get_device, load_config,
                    plot_confusion_matrix, plot_misclassified_samples,
                    plot_training_curves, project_path, save_checkpoint,
                    save_metrics, set_seed)


def _run_accuracy(output: torch.Tensor, target: torch.Tensor) -> int:
    return int((output.argmax(dim=1) == target).sum().item())


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                    optimizer: Optimizer, device: torch.device) -> tuple[float, float]:
    """Train for one epoch using the explicit forward/backward update steps."""
    model.train()
    total_loss = total_correct = total_items = 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * targets.size(0)
        total_correct += _run_accuracy(output, targets)
        total_items += targets.size(0)
    return total_loss / total_items, total_correct / total_items


def validate_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                       device: torch.device) -> tuple[float, float]:
    """Evaluate one validation epoch without computing gradients."""
    model.eval()
    total_loss = total_correct = total_items = 0
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            output = model(images)
            loss = criterion(output, targets)
            total_loss += loss.item() * targets.size(0)
            total_correct += _run_accuracy(output, targets)
            total_items += targets.size(0)
    return total_loss / total_items, total_correct / total_items


def train_model(model: nn.Module, loaders: dict[str, DataLoader], config: dict[str, Any],
                device: torch.device) -> dict[str, list[float]]:
    """Train a model and return loss/accuracy history for later plotting."""
    criterion = nn.CrossEntropyLoss()
    training = config.get("training", {})
    optimizer_setting = config.get("optimizer", "adam")
    optimizer_name = (optimizer_setting.get("name", "adam")
                      if isinstance(optimizer_setting, dict) else optimizer_setting).lower()
    learning_rate = float(training.get("learning_rate", config.get("learning_rate", 1e-3)))
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    else:
        raise ValueError("Unsupported optimizer. Choose adam or sgd.")
    model.to(device)
    history = {"train_loss": [], "validation_loss": [], "train_accuracy": [], "validation_accuracy": []}
    for _ in range(int(training.get("epochs", config.get("epochs", 10)))):
        train_loss, train_accuracy = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        validation_loss, validation_accuracy = validate_one_epoch(
            model, loaders["validation"], criterion, device
        )
        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation_loss)
        history["train_accuracy"].append(train_accuracy)
        history["validation_accuracy"].append(validation_accuracy)
    return history


def parse_args() -> argparse.Namespace:
    """Parse command-line overrides used locally and in Colab."""
    parser = argparse.ArgumentParser(description="Train one E1 classifier.")
    parser.add_argument("--model", choices=["softmax", "mlp", "cnn"], required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    return parser.parse_args()


def main() -> None:
    """Train, evaluate, and save artifacts for one selected classifier."""
    args = parse_args()
    project_root = Path.cwd()
    config = load_config(args.config)
    config.setdefault("training", {})
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr
    set_seed(int(config.get("seed", 42)))
    device = get_device()
    bundle = load_datasets(config)
    loaders = create_dataloaders(bundle, config)
    model = get_model(args.model, config, bundle.input_shape, bundle.num_classes)
    start = time.perf_counter()
    history = train_model(model, loaders, config, device)
    training_time = time.perf_counter() - start
    evaluation = evaluate_model(model, loaders["test"], device)
    output = config["output"]
    metrics_dir = project_path(output["metrics_dir"], project_root)
    figures_dir = project_path(output["figures_dir"], project_root)
    checkpoints_dir = project_path(output["checkpoints_dir"], project_root)
    save_metrics({"model": args.model, "history": history,
                  "test_accuracy": evaluation["accuracy"],
                  "num_parameters": count_trainable_parameters(model),
                  "best_val_accuracy": max(history["validation_accuracy"]),
                  "training_time": training_time}, metrics_dir / f"{args.model}.json")
    save_checkpoint(model, checkpoints_dir / f"{args.model}.pt")
    plot_training_curves(history, figures_dir / f"{args.model}_learning_curves.png")
    plot_confusion_matrix(evaluation["confusion_matrix"], bundle.class_names,
                          figures_dir / f"{args.model}_confusion_matrix.png")
    plot_misclassified_samples(misclassified_samples(evaluation), bundle.class_names,
                               figures_dir / f"{args.model}_misclassified.png")
    _update_comparison(metrics_dir / "model_comparison.csv", args.model, evaluation,
                       history, training_time)
    print(f"Completed {args.model} on {device}; test accuracy: {evaluation['accuracy']:.4f}")


def _update_comparison(path: Path, model_name: str, evaluation: dict[str, Any],
                       history: dict[str, list[float]], training_time: float) -> None:
    """Upsert one real experiment result into the comparison CSV."""
    rows: dict[str, dict[str, Any]] = {}
    if path.exists():
        with path.open(newline="", encoding="utf-8") as file:
            rows = {row["model"]: row for row in csv.DictReader(file)}
    rows[model_name] = {"model": model_name, "test_accuracy": evaluation["accuracy"],
                       "num_parameters": evaluation["trainable_parameters"],
                       "best_val_accuracy": max(history["validation_accuracy"]),
                       "training_time": training_time}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["model", "test_accuracy", "num_parameters", "best_val_accuracy", "training_time"])
        writer.writeheader()
        writer.writerows(rows.values())


if __name__ == "__main__":
    main()
