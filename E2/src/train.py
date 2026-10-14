"""Training loop tự viết cho E2."""

from __future__ import annotations

import argparse
import csv
import shutil
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .data import create_dataloaders, load_datasets
from .evaluate import accuracy_count, evaluate_model
from .model import build_model
from .utils import (count_trainable_parameters, get_device, load_config,
                    plot_confusion_matrix, plot_training_curves, resolve_path,
                    save_checkpoint, save_json, set_seed)


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                    optimizer: Optimizer, device: torch.device) -> tuple[float, float]:
    """Một epoch huấn luyện: forward -> loss -> backward() -> optimizer.step()."""
    model.train()
    total_loss = total_correct = total_items = 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * targets.size(0)
        total_correct += accuracy_count(logits, targets)
        total_items += targets.size(0)
    return total_loss / total_items, total_correct / total_items


def validate_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                       device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = total_correct = total_items = 0
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            loss = criterion(logits, targets)
            total_loss += loss.item() * targets.size(0)
            total_correct += accuracy_count(logits, targets)
            total_items += targets.size(0)
    return total_loss / total_items, total_correct / total_items


def make_optimizer(model: nn.Module, config: dict[str, Any]) -> Optimizer:
    optimizer_config = config.get("optimizer", {})
    name = optimizer_config.get("name", "adam").lower()
    lr = float(config.get("training", {}).get("learning_rate", 1e-3))
    weight_decay = float(optimizer_config.get("weight_decay", 0.0))
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError("optimizer.name must be adam, adamw, or sgd")


def train_model(model: nn.Module, loaders: dict[str, DataLoader], config: dict[str, Any],
                device: torch.device, checkpoint_path: Path | None = None,
                resume_path: Path | None = None) -> tuple[dict[str, list[float]], float]:
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model, config)
    model.to(device)
    start_epoch = 0
    best_validation_accuracy = -1.0
    if resume_path is not None and resume_path.exists():
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0))
        best_validation_accuracy = float(checkpoint.get("best_validation_accuracy", -1.0))

    history = {"train_loss": [], "validation_loss": [], "train_accuracy": [], "validation_accuracy": []}
    epochs = int(config.get("training", {}).get("epochs", 5))
    start_time = time.perf_counter()
    for epoch in range(start_epoch, epochs):
        train_loss, train_accuracy = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        validation_loss, validation_accuracy = validate_one_epoch(model, loaders["validation"], criterion, device)
        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation_loss)
        history["train_accuracy"].append(train_accuracy)
        history["validation_accuracy"].append(validation_accuracy)
        if checkpoint_path is not None and validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            save_checkpoint(model, optimizer, epoch + 1, best_validation_accuracy, checkpoint_path, config)
        print(
            f"epoch={epoch + 1}/{epochs} train_loss={train_loss:.4f} "
            f"val_loss={validation_loss:.4f} train_acc={train_accuracy:.4f} val_acc={validation_accuracy:.4f}"
        )
    training_time = time.perf_counter() - start_time
    return history, training_time


def update_comparison(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: dict[str, dict[str, Any]] = {}
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as file:
            rows = {item["experiment_name"]: item for item in csv.DictReader(file)}
    rows[row["experiment_name"]] = row
    fieldnames = [
        "experiment_name", "attention_type", "tokenizer", "test_accuracy",
        "best_validation_accuracy", "num_parameters", "training_time", "inference_time",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one E2 Transformer image classifier.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--experiment-name")
    parser.add_argument("--attention", choices=["custom", "torch"])
    parser.add_argument("--tokenizer", choices=["patch", "row", "cnn_stem"])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--resume", type=str)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path.cwd()
    config = load_config(args.config)
    config.setdefault("training", {})
    config.setdefault("model", {})
    if args.attention:
        config["model"]["attention_type"] = args.attention
    if args.tokenizer:
        config["model"]["tokenizer"] = args.tokenizer
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr

    experiment_name = args.experiment_name or f"{config['model']['attention_type']}_{config['model']['tokenizer']}"
    set_seed(int(config.get("seed", 42)))
    device = get_device()
    bundle = load_datasets(config, project_root)
    loaders = create_dataloaders(bundle, config)
    model = build_model(config, bundle.input_shape, bundle.num_classes)

    output = config.get("output", {})
    metrics_dir = resolve_path(output.get("metrics_dir", "outputs/metrics"), project_root)
    figures_dir = resolve_path(output.get("figures_dir", "outputs/figures"), project_root)
    checkpoints_dir = resolve_path(output.get("checkpoints_dir", "outputs/checkpoints"), project_root)
    config_dir = resolve_path(output.get("configs_dir", "outputs/configs"), project_root)
    checkpoint_path = checkpoints_dir / f"{experiment_name}.pt"
    resume_path = Path(args.resume) if args.resume else None

    save_json(config, config_dir / f"{experiment_name}.json")
    history, training_time = train_model(model, loaders, config, device, checkpoint_path, resume_path)
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    evaluation = evaluate_model(model, loaders["test"], device, bundle.num_classes)

    metrics = {
        "experiment_name": experiment_name,
        "dataset": config.get("dataset", {}).get("name", "fashion_mnist"),
        "device": str(device),
        "model": config["model"],
        "history": history,
        "test": evaluation,
        "best_validation_accuracy": max(history["validation_accuracy"]) if history["validation_accuracy"] else None,
        "training_time": training_time,
        "num_parameters": count_trainable_parameters(model),
    }
    save_json(metrics, metrics_dir / f"{experiment_name}.json")
    plot_training_curves(history, figures_dir / f"{experiment_name}_learning_curves.png")
    plot_confusion_matrix(evaluation["confusion_matrix"], bundle.class_names, figures_dir / f"{experiment_name}_confusion_matrix.png")
    update_comparison(metrics_dir / "comparison.csv", {
        "experiment_name": experiment_name,
        "attention_type": config["model"]["attention_type"],
        "tokenizer": config["model"]["tokenizer"],
        "test_accuracy": evaluation["accuracy"],
        "best_validation_accuracy": metrics["best_validation_accuracy"],
        "num_parameters": metrics["num_parameters"],
        "training_time": training_time,
        "inference_time": evaluation["inference_time"],
    })
    if output.get("copy_to_dir"):
        copy_dir = resolve_path(output["copy_to_dir"], project_root)
        copy_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(metrics_dir / f"{experiment_name}.json", copy_dir / f"{experiment_name}.json")
    print(f"Completed {experiment_name} on {device}; test accuracy: {evaluation['accuracy']:.4f}")


if __name__ == "__main__":
    main()
