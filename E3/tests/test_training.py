"""Lightweight tests for the reusable E3 training and evaluation pipeline."""

import csv
import json
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from E3.src.evaluate import evaluate_checkpoint, evaluate_model
from E3.src.models import get_model
from E3.src.train import train_model, validate_one_epoch
from E3.src.utils import load_checkpoint, save_experiment_results, save_history


@pytest.fixture
def loaders() -> dict[str, DataLoader]:
    generator = torch.Generator().manual_seed(7)
    images = torch.randn(16, 1, 4, 4, generator=generator)
    labels = torch.arange(16) % 3
    return {
        "train": DataLoader(TensorDataset(images[:8], labels[:8]), batch_size=4),
        "validation": DataLoader(TensorDataset(images[8:12], labels[8:12]), batch_size=2),
        "test": DataLoader(TensorDataset(images[12:], labels[12:]), batch_size=2),
    }


@pytest.fixture
def config() -> dict:
    return {
        "seed": 42,
        "input_size": 4,
        "hidden_size": 5,
        "num_layers": 1,
        "dropout": 0.0,
        "num_classes": 3,
        "training": {
            "batch_size": 4,
            "epochs": 1,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "weight_decay": 0.0,
        },
    }


@pytest.mark.parametrize("model_name", ["lstm", "gru"])
def test_cpu_training_checkpoint_evaluation_and_serialization(
    model_name: str,
    loaders: dict[str, DataLoader],
    config: dict,
    tmp_path: Path,
) -> None:
    model = get_model(model_name, config)
    checkpoint_path = tmp_path / "checkpoints" / f"{model_name}_best.pth"

    history = train_model(
        model, model_name, loaders, config, checkpoint_path, device=torch.device("cpu")
    )
    assert len(history["train_loss"]) == len(history["validation_loss"]) == 1
    assert len(history["train_accuracy"]) == len(history["validation_accuracy"]) == 1
    assert history["training_time"] >= 0
    assert history["best_val_accuracy"] == history["validation_accuracy"][0]

    checkpoint = load_checkpoint(checkpoint_path, torch.device("cpu"))
    assert {
        "model_state_dict",
        "optimizer_state_dict",
        "model_name",
        "epoch",
        "best_val_accuracy",
        "config",
        "seed",
    }.issubset(checkpoint)
    assert checkpoint["model_name"] == model_name
    assert checkpoint["epoch"] == 1
    assert checkpoint["seed"] == 42

    validation_metrics = validate_one_epoch(
        model, loaders["validation"], torch.nn.CrossEntropyLoss(), torch.device("cpu")
    )
    assert 0 <= validation_metrics[1] <= 1
    assert validation_metrics[0] >= 0

    metrics = evaluate_checkpoint(
        get_model(model_name, config), checkpoint_path, loaders["test"], torch.device("cpu")
    )
    assert metrics["model_name"] == model_name
    assert metrics["num_parameters"] > 0
    assert metrics["test_loss"] >= 0
    assert 0 <= metrics["test_accuracy"] <= 1
    assert metrics["best_val_accuracy"] == history["best_val_accuracy"]
    assert metrics["training_time"] == history["training_time"]

    direct_metrics = evaluate_model(model, loaders["test"], torch.device("cpu"))
    assert direct_metrics["num_parameters"] == metrics["num_parameters"]

    history_csv = tmp_path / "metrics" / f"{model_name}_history.csv"
    save_history(history, history_csv)
    with history_csv.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 1
    assert int(rows[0]["epoch"]) == 1
    history_json = tmp_path / "metrics" / f"{model_name}_history.json"
    save_history(history, history_json)
    assert json.loads(history_json.read_text(encoding="utf-8"))["best_epoch"] == 1

    results_csv = tmp_path / "metrics" / "results.csv"
    results_json = tmp_path / "metrics" / "results.json"
    save_experiment_results(metrics, results_csv, results_json)
    with results_csv.open(newline="", encoding="utf-8") as file:
        assert len(list(csv.DictReader(file))) == 1
    serialized = json.loads(results_json.read_text(encoding="utf-8"))
    assert serialized[model_name]["test_accuracy"] == metrics["test_accuracy"]


def test_validation_does_not_update_parameters(
    loaders: dict[str, DataLoader], config: dict
) -> None:
    model = get_model("gru", config)
    original = {key: value.clone() for key, value in model.state_dict().items()}
    validate_one_epoch(
        model, loaders["validation"], torch.nn.CrossEntropyLoss(), torch.device("cpu")
    )
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in original.items())
