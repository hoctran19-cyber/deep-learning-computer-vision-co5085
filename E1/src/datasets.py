"""Dataset and DataLoader helpers for E1.

The default dataset is Fashion-MNIST. The same split indices can be reused for
all three models to keep comparisons fair.
"""

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms


DATASET_NAMES = {"fashion_mnist", "mnist", "cifar10"}


@dataclass
class DatasetBundle:
    """Train, validation, and test datasets plus basic image metadata."""

    train: Dataset
    validation: Dataset
    test: Dataset
    input_shape: tuple[int, int, int]
    num_classes: int
    class_names: list[str]


def _dataset_class(name: str) -> type:
    """Return the torchvision dataset class for a configured name."""
    normalized = name.lower()
    if normalized not in DATASET_NAMES:
        raise ValueError(f"Unsupported dataset '{name}'. Choose from {sorted(DATASET_NAMES)}.")
    return {
        "fashion_mnist": datasets.FashionMNIST,
        "mnist": datasets.MNIST,
        "cifar10": datasets.CIFAR10,
    }[normalized]


def _metadata(name: str) -> tuple[tuple[int, int, int], int, list[str]]:
    """Return expected shape, class count, and labels for a dataset."""
    if name.lower() == "cifar10":
        return (3, 32, 32), 10, [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck",
        ]
    classes = (
        ["t-shirt/top", "trouser", "pullover", "dress", "coat",
         "sandal", "shirt", "sneaker", "bag", "ankle boot"]
        if name.lower() == "fashion_mnist"
        else [str(index) for index in range(10)]
    )
    return (1, 28, 28), 10, classes


def _transform(name: str) -> transforms.Compose:
    """Create the common tensor transform for the selected dataset."""
    normalize = ((0.5,), (0.5,)) if name.lower() != "cifar10" else (
        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    )
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(*normalize)])


def load_datasets(config: dict[str, Any]) -> DatasetBundle:
    """Load train/test datasets and make a deterministic train/validation split.

    Setting ``dataset.download`` to true is required before torchvision will
    download missing files. This function is not called during project setup.
    """
    dataset_config = config["dataset"]
    name = dataset_config.get("name", "fashion_mnist").lower()
    dataset_class = _dataset_class(name)
    transform = _transform(name)
    root = dataset_config.get("data_dir", dataset_config.get("root", "data"))
    download = bool(dataset_config.get("download", False))

    full_train = dataset_class(root=root, train=True, transform=transform, download=download)
    test = dataset_class(root=root, train=False, transform=transform, download=download)
    validation_size = int(len(full_train) * float(
        config.get("experiment", {}).get("validation_split",
                    config.get("validation_split", 0.1))))
    train_size = len(full_train) - validation_size
    generator = torch.Generator().manual_seed(int(config.get("seed", config.get("random_seed", 42))))
    train, validation = random_split(full_train, [train_size, validation_size], generator=generator)
    shape, classes, class_names = _metadata(name)
    return DatasetBundle(train, validation, test, shape, classes, class_names)


def create_dataloaders(bundle: DatasetBundle, config: dict[str, Any]) -> dict[str, DataLoader]:
    """Create loaders for the three dataset splits."""
    batch_size = int(config.get("training", {}).get("batch_size", config.get("batch_size", 128)))
    workers = int(config["dataset"].get("num_workers", 2))
    return {
        "train": DataLoader(bundle.train, batch_size=batch_size, shuffle=True, num_workers=workers),
        "validation": DataLoader(bundle.validation, batch_size=batch_size, shuffle=False, num_workers=workers),
        "test": DataLoader(bundle.test, batch_size=batch_size, shuffle=False, num_workers=workers),
    }
