"""Fashion-MNIST loaders shared with the E1 baseline pipeline."""

from typing import Any

from torch.utils.data import DataLoader
from torch import Tensor

from E1.src.datasets import DatasetBundle
from E1.src.datasets import create_dataloaders as _create_dataloaders
from E1.src.datasets import load_datasets as _load_datasets


def load_datasets(config: dict[str, Any]) -> DatasetBundle:
    """Load Fashion-MNIST using E1's transform and seeded train/validation split."""
    return _load_datasets(config)


def create_dataloaders(
    bundle: DatasetBundle, config: dict[str, Any]
) -> dict[str, DataLoader]:
    """Create loaders using E1's shuffle and batching behavior."""
    return _create_dataloaders(bundle, config)


def image_to_row_sequence(images: Tensor) -> Tensor:
    """Convert grayscale images from (batch, 1, height, width) to row sequences.

    The resulting tensor has shape (batch, height, width), with rows kept in
    their original top-to-bottom order.
    """
    if images.ndim != 4 or images.shape[1] != 1:
        raise ValueError("Expected grayscale images with shape (batch, 1, height, width).")
    if images.shape[2] == 0 or images.shape[3] == 0:
        raise ValueError("Image height and width must be positive.")
    return images.squeeze(1)
