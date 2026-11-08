"""Smoke-test Food101Dataset and its DataLoader using an extracted dataset."""

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from torch.utils.data import DataLoader


EXPECTED_COUNTS = {"train": 68175, "validation": 7575, "test": 25250}


class SmokeTestError(Exception):
    """A DataLoader smoke-test assertion failed."""


def require(condition: bool, message: str) -> None:
    """Raise a descriptive error when a test condition is false."""
    if not condition:
        raise SmokeTestError(message)


def check_batch(batch: object, expected_size: int) -> None:
    """Check collated image and label tensors, including their values."""
    import torch

    require(
        isinstance(batch, (list, tuple)) and len(batch) == 2,
        "batch must contain images and labels",
    )
    images, labels = batch
    require(isinstance(images, torch.Tensor), "images must be a tensor")
    require(isinstance(labels, torch.Tensor), "labels must be a tensor")
    require(
        images.ndim == 4
        and images.shape[0] == expected_size
        and images.shape[1] == 3,
        f"invalid image batch shape: {tuple(images.shape)}",
    )
    require(
        images.shape[2] > 0 and images.shape[3] > 0,
        "image dimensions must be positive",
    )
    require(
        images.dtype == torch.float32,
        f"expected float32 images, got {images.dtype}",
    )
    require(
        bool(torch.isfinite(images).all())
        and bool(((images >= 0) & (images <= 1)).all()),
        "image values must be finite and in [0, 1]",
    )
    require(
        labels.ndim == 1 and labels.shape[0] == expected_size,
        f"invalid label batch shape: {tuple(labels.shape)}",
    )
    require(
        labels.dtype == torch.int64,
        f"expected int64 labels, got {labels.dtype}",
    )
    require(
        bool(((labels >= 0) & (labels <= 100)).all()),
        "labels must be in [0, 100]",
    )


def check_batch_sampler(
    loader: "DataLoader", sample_count: int, batch_size: int
) -> list[int]:
    """Check an index-only epoch and return its actual last batch indices."""
    require(not loader.drop_last, "drop_last must be False")
    expected_batches = (sample_count + batch_size - 1) // batch_size
    require(
        len(loader) == expected_batches,
        f"expected {expected_batches} batches, got {len(loader)}",
    )
    seen = bytearray(sample_count)
    batch_count = 0
    last_batch: list[int] = []
    for indices in loader.batch_sampler:
        batch_count += 1
        last_batch = list(indices)
        require(
            0 < len(last_batch) <= batch_size,
            f"invalid batch size {len(last_batch)}",
        )
        for index in last_batch:
            require(
                0 <= index < sample_count,
                f"sample index {index} out of range",
            )
            require(not seen[index], f"duplicate sample index {index}")
            seen[index] = 1
    require(
        batch_count == expected_batches,
        f"sampler yielded {batch_count} batches, expected {expected_batches}",
    )
    require(
        sum(seen) == sample_count,
        f"sampler omitted {sample_count - sum(seen)} samples",
    )
    expected_last = sample_count % batch_size or batch_size
    require(
        len(last_batch) == expected_last,
        f"last batch has {len(last_batch)} samples, expected {expected_last}",
    )
    return last_batch


def run_smoke_tests(
    dataset_root: Path, manifest: Path, batch_size: int
) -> bool:
    """Check each split, reading one sample and the first and last batches."""
    from PIL import Image
    from torch.utils.data import RandomSampler, SequentialSampler
    from torchvision import transforms

    from dataset import Food101Dataset, create_dataloader

    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    passed = True
    for split, expected_count in EXPECTED_COUNTS.items():
        try:
            dataset = Food101Dataset(dataset_root, manifest, split=split)
            require(
                len(dataset) == expected_count,
                f"expected {expected_count:,} samples, got {len(dataset):,}",
            )
            print(f"[PASS] {split}: {len(dataset):,} samples")

            image, label = dataset[0]
            require(
                isinstance(image, Image.Image) and image.mode == "RGB",
                "sample image must be a PIL RGB image",
            )
            require(
                isinstance(label, int)
                and not isinstance(label, bool)
                and 0 <= label <= 100,
                f"invalid sample label: {label!r}",
            )
            print(f"[PASS] {split}: RGB sample and label")

            dataset.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
            loader = create_dataloader(
                dataset, batch_size=batch_size, num_workers=0, pin_memory=False
            )
            expected_sampler = (
                RandomSampler if split == "train" else SequentialSampler
            )
            require(
                isinstance(loader.sampler, expected_sampler),
                f"expected {expected_sampler.__name__}, "
                f"got {type(loader.sampler).__name__}",
            )
            if split == "train":
                require(
                    not loader.sampler.replacement,
                    "train sampler must sample without replacement",
                )
            print(
                f"[PASS] {split}: "
                f"shuffle {'enabled' if split == 'train' else 'disabled'}"
            )

            check_batch(next(iter(loader)), min(batch_size, len(dataset)))
            print(f"[PASS] {split}: first batch shape, dtype and values")

            last_indices = check_batch_sampler(
                loader, len(dataset), batch_size
            )
            last_batch = loader.collate_fn(
                [dataset[index] for index in last_indices]
            )
            check_batch(last_batch, len(last_indices))
            print(
                f"[PASS] {split}: drop_last=False, full index coverage "
                f"and last batch ({len(last_indices)} samples)"
            )
        except (
            OSError,
            ValueError,
            TypeError,
            RuntimeError,
            StopIteration,
            IndexError,
            SmokeTestError,
        ) as error:
            print(
                f"[FAIL] {split}: {type(error).__name__}: {error}",
                file=sys.stderr,
            )
            passed = False
    return passed


def main() -> int:
    """Parse arguments and report a nonzero exit status for failed tests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path, required=True,
        help="Extracted Food-101 root with meta/ and images/",
    )
    parser.add_argument(
        "--manifest", type=Path, required=True,
        help="Path to the Food-101 CSV manifest",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size (default: 8)"
    )
    args = parser.parse_args()
    try:
        passed = run_smoke_tests(
            args.dataset_root, args.manifest, args.batch_size
        )
    except (ImportError, OSError, ValueError, RuntimeError) as error:
        print(
            f"[FAIL] Setup: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        passed = False
    print("ALL TESTS PASSED" if passed else "TESTS FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
