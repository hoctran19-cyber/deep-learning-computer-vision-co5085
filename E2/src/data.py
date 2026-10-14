"""Dataset và DataLoader cho E2, giữ quy ước tương thích với E1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms


DATASET_NAMES = {"fashion_mnist", "mnist", "cifar10"}


@dataclass
class DatasetBundle:
    train: Dataset
    validation: Dataset
    test: Dataset
    input_shape: tuple[int, int, int]
    num_classes: int
    class_names: list[str]


def _dataset_class(name: str) -> type:
    normalized = name.lower()
    if normalized not in DATASET_NAMES:
        raise ValueError(f"Unsupported dataset '{name}'. Choose from {sorted(DATASET_NAMES)}")
    return {"fashion_mnist": datasets.FashionMNIST, "mnist": datasets.MNIST, "cifar10": datasets.CIFAR10}[normalized]


def dataset_metadata(name: str) -> tuple[tuple[int, int, int], int, list[str]]:
    if name.lower() == "cifar10":
        return (3, 32, 32), 10, [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck",
        ]
    classes = [
        "t-shirt/top", "trouser", "pullover", "dress", "coat",
        "sandal", "shirt", "sneaker", "bag", "ankle boot",
    ] if name.lower() == "fashion_mnist" else [str(i) for i in range(10)]
    return (1, 28, 28), 10, classes


def _transform(name: str) -> transforms.Compose:
    normalize = ((0.5,), (0.5,)) if name.lower() != "cifar10" else ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(*normalize)])


def load_datasets(config: dict[str, Any], project_root: Path | None = None) -> DatasetBundle:
    dataset_config = config.get("dataset", {})
    name = dataset_config.get("name", "fashion_mnist").lower()
    data_dir = Path(dataset_config.get("data_dir", "data"))
    if not data_dir.is_absolute() and project_root is not None:
        data_dir = project_root / data_dir
    dataset_class = _dataset_class(name)
    transform = _transform(name)
    full_train = dataset_class(root=data_dir, train=True, transform=transform, download=bool(dataset_config.get("download", False)))
    test = dataset_class(root=data_dir, train=False, transform=transform, download=bool(dataset_config.get("download", False)))
    validation_split = float(config.get("experiment", {}).get("validation_split", 0.1))
    validation_size = int(len(full_train) * validation_split)
    train_size = len(full_train) - validation_size
    generator = torch.Generator().manual_seed(int(config.get("seed", 42)))
    train, validation = random_split(full_train, [train_size, validation_size], generator=generator)
    shape, num_classes, class_names = dataset_metadata(name)
    return DatasetBundle(train, validation, test, shape, num_classes, class_names)


def create_dataloaders(bundle: DatasetBundle, config: dict[str, Any]) -> dict[str, DataLoader]:
    training_config = config.get("training", {})
    dataset_config = config.get("dataset", {})
    batch_size = int(training_config.get("batch_size", 128))
    workers = int(dataset_config.get("num_workers", 2))
    return {
        "train": DataLoader(bundle.train, batch_size=batch_size, shuffle=True, num_workers=workers),
        "validation": DataLoader(bundle.validation, batch_size=batch_size, shuffle=False, num_workers=workers),
        "test": DataLoader(bundle.test, batch_size=batch_size, shuffle=False, num_workers=workers),
    }
