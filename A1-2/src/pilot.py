"""Chuẩn bị tập con pilot từ train/validation chính thức của Food-101."""

import json
import random
from collections import Counter
from pathlib import Path
from typing import Callable

from PIL import Image
from torch import Tensor
from torch.utils.data import Subset

if __package__:
    from .dataset import Food101Dataset
else:
    from dataset import Food101Dataset


class PilotSubset(Subset):
    """Subset giữ tên split cho create_dataloader và fit."""

    def __init__(self, dataset: Food101Dataset, indices: list[int]) -> None:
        super().__init__(dataset, indices)
        self.split = dataset.split


def prepare_pilot_datasets(
    dataset_root: Path,
    manifest: Path,
    train_transform: Callable[[Image.Image], Tensor],
    eval_transform: Callable[[Image.Image], Tensor],
    seed: int = 42,
) -> tuple[PilotSubset, PilotSubset]:
    """Kiểm tra split, đối chiếu train.json và lấy 5/2 ảnh cho mỗi lớp.

    Chỉ khởi tạo dataset train và validation, không nạp test.json hay ảnh test.
    Food101Dataset vẫn kiểm tra cấu trúc và trùng lặp của toàn bộ manifest.
    """
    train = Food101Dataset(dataset_root, manifest, "train", train_transform)
    validation = Food101Dataset(
        dataset_root, manifest, "validation", eval_transform
    )
    if train.classes != validation.classes or len(train.classes) != 101:
        raise ValueError("train/validation class mappings must contain 101 classes")
    root = dataset_root.resolve()
    actual: set[str] = set()
    for dataset, expected in ((train, 675), (validation, 75)):
        counts = Counter(sample["class_name"] for sample in dataset.samples)
        if set(counts) != set(dataset.classes) or any(
            counts[name] != expected for name in dataset.classes
        ):
            raise ValueError(f"{dataset.split}: invalid per-class image counts")
        for sample in dataset.samples:
            relative = sample["image_path"].relative_to(root).as_posix()
            if relative in actual:
                raise ValueError(f"duplicate train/validation image: {relative}")
            actual.add(relative)
    with (root / "meta" / "train.json").open(encoding="utf-8") as file:
        official = json.load(file)
    if not isinstance(official, dict) or set(official) != set(train.classes):
        raise ValueError("train.json classes do not match classes.txt")
    expected_paths: set[str] = set()
    for name in train.classes:
        image_ids = official[name]
        if not isinstance(image_ids, list) or len(image_ids) != 750:
            raise ValueError(f"train.json: expected 750 images for {name}")
        for image_id in image_ids:
            if not isinstance(image_id, str):
                raise ValueError(f"train.json: invalid image ID for {name}")
            relative = f"images/{image_id}"
            if not relative.endswith(".jpg"):
                relative += ".jpg"
            parts = relative.split("/")
            if (
                len(parts) != 3 or parts[1] != name
                or not parts[2] or parts[2].startswith(".")
                or "\\" in relative or parts[2] == ".jpg"
            ):
                raise ValueError(f"train.json: invalid image ID {image_id!r}")
            if relative in expected_paths:
                raise ValueError(f"train.json: duplicate image ID {image_id!r}")
            expected_paths.add(relative)
    if actual != expected_paths:
        raise ValueError(
            "train/validation membership does not match train.json: "
            f"{len(expected_paths - actual)} missing, "
            f"{len(actual - expected_paths)} unexpected"
        )

    rng = random.Random(seed)
    result = []
    for dataset, per_class in ((train, 5), (validation, 2)):
        by_class: dict[str, list[int]] = {name: [] for name in dataset.classes}
        for index, sample in enumerate(dataset.samples):
            by_class[sample["class_name"]].append(index)
        indices = [
            index
            for name in dataset.classes
            for index in rng.sample(by_class[name], per_class)
        ]
        result.append(PilotSubset(dataset, indices))
    return result[0], result[1]
