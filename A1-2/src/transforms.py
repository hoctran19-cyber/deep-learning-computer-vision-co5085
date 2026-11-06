"""Tiền xử lý ảnh Food-101 theo protocol A1-1."""

import math
from collections.abc import Sequence

from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _validate(
    resize_short_edge: int, crop_size: int,
    mean: Sequence[float], std: Sequence[float],
) -> None:
    """Kiểm tra tham số trước khi tạo pipeline."""
    for name, value in (("resize_short_edge", resize_short_edge), ("crop_size", crop_size)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if crop_size > resize_short_edge:
        raise ValueError("crop_size must not exceed resize_short_edge")
    if len(mean) != 3 or len(std) != 3:
        raise ValueError("mean and std must have three channels")
    if not all(math.isfinite(value) for value in (*mean, *std)):
        raise ValueError("mean and std must be finite")
    if any(value <= 0 for value in std):
        raise ValueError("std values must be positive")


def build_train_transform(
    resize_short_edge: int = 256,
    crop_size: int = 224,
    horizontal_flip_probability: float = 0.5,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> transforms.Compose:
    """Resize, random crop/flip và chuẩn hóa ảnh RGB để huấn luyện."""
    _validate(resize_short_edge, crop_size, mean, std)
    if not 0 <= horizontal_flip_probability <= 1:
        raise ValueError("horizontal_flip_probability must be in [0, 1]")
    return transforms.Compose([
        transforms.Resize(resize_short_edge),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(p=horizontal_flip_probability),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def build_eval_transform(
    resize_short_edge: int = 256,
    crop_size: int = 224,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> transforms.Compose:
    """Resize, center crop và chuẩn hóa ảnh RGB để validation."""
    _validate(resize_short_edge, crop_size, mean, std)
    return transforms.Compose([
        transforms.Resize(resize_short_edge),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
