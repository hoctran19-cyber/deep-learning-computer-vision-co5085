
"""Generate reproducible Food-101 train/validation/test manifests.

Protocol:
- Official train: 750 images per class.
- Train/validation: 90/10 stratified split.
- Official test: preserved without modification.
- Random seed: 42.
"""

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path, PurePosixPath


SEED = 42
NUM_CLASSES = 101
OFFICIAL_TRAIN_PER_CLASS = 750
OFFICIAL_TEST_PER_CLASS = 250
VALIDATION_RATIO = 0.10

MANIFEST_COLUMNS = [
    "image_path",
    "class_name",
    "label",
    "split",
]


def load_classes(dataset_root: Path) -> list[str]:
    """Load and validate Food-101 class mapping."""

    classes_file = dataset_root / "meta" / "classes.txt"

    if not classes_file.is_file():
        raise FileNotFoundError(
            f"Classes file not found: {classes_file}"
        )

    with open(classes_file, encoding="utf-8") as file:
        classes = [
            line.strip()
            for line in file
            if line.strip()
        ]

    if len(classes) != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} classes, got {len(classes)}"
        )

    if len(set(classes)) != len(classes):
        raise ValueError("Duplicate class names detected")

    if len(classes) != len(set(classes)):
        raise ValueError("classes.txt contains duplicate classes")

    return classes


def load_metadata(
    metadata_path: Path,
    classes: list[str],
    expected_per_class: int,
) -> dict[str, list[str]]:
    """Load and validate official Food-101 metadata JSON."""

    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}"
        )

    with open(metadata_path, encoding="utf-8") as file:
        metadata = json.load(file)

    if not isinstance(metadata, dict):
        raise ValueError("Metadata must be a JSON dictionary")

    if set(metadata.keys()) != set(classes):
        raise ValueError(
            f"Metadata classes do not match classes.txt: "
            f"{metadata_path}"
        )

    for class_name in classes:
        image_ids = metadata[class_name]

        if not isinstance(image_ids, list):
            raise ValueError(
                f"Invalid metadata list for {class_name}"
            )

        if len(image_ids) != expected_per_class:
            raise ValueError(
                f"{class_name}: expected {expected_per_class} "
                f"images, got {len(image_ids)}"
            )

        for image_id in image_ids:
            if not isinstance(image_id, str) or not image_id.strip():
                raise ValueError(
                    f"Invalid image ID in class {class_name}"
                )

            relative_path = PurePosixPath(image_id)

            if (
                relative_path.is_absolute()
                or ".." in relative_path.parts
                or len(relative_path.parts) != 2
                or relative_path.parts[0] != class_name
            ):
                raise ValueError(
                    f"Invalid image ID: {image_id}"
                )

            if relative_path.suffix not in {"", ".jpg"}:
                raise ValueError(
                    f"Unexpected image extension: {image_id}"
                )

    return metadata


def build_manifest(
    classes: list[str],
    train_metadata: dict[str, list[str]],
    test_metadata: dict[str, list[str]],
    seed: int = SEED,
) -> list[dict]:
    """Build stratified train/validation/test records."""

    rng = random.Random(seed)
    class_to_idx = {
        class_name: idx
        for idx, class_name in enumerate(classes)
    }

    records = []
    seen_paths = set()

    for class_name in classes:
        label = class_to_idx[class_name]

        # Split only the official training metadata.
        train_ids = list(train_metadata[class_name])
        rng.shuffle(train_ids)

        validation_count = int(
            OFFICIAL_TRAIN_PER_CLASS * VALIDATION_RATIO
        )

        validation_ids = train_ids[:validation_count]
        training_ids = train_ids[validation_count:]

        # Preserve official test metadata without resampling.
        split_groups = [
            ("train", training_ids),
            ("validation", validation_ids),
            ("test", list(test_metadata[class_name])),
        ]

        for split_name, image_ids in split_groups:
            for image_id in sorted(image_ids):
                relative_path = PurePosixPath(
                    "images"
                ) / PurePosixPath(image_id)

                if relative_path.suffix == "":
                    relative_path = relative_path.with_suffix(".jpg")

                image_path = relative_path.as_posix()

                if image_path in seen_paths:
                    raise ValueError(
                        f"Duplicate image path: {image_path}"
                    )

                seen_paths.add(image_path)

                records.append({
                    "image_path": image_path,
                    "class_name": class_name,
                    "label": label,
                    "split": split_name,
                })

    return records


def validate_manifest(records: list[dict], classes: list[str]) -> None:
    """Validate split counts and class distribution."""

    expected_counts = {
        "train": 675,
        "validation": 75,
        "test": 250,
    }

    class_to_idx = {
        class_name: idx
        for idx, class_name in enumerate(classes)
    }

    class_split_counts = {
        split: Counter()
        for split in expected_counts
    }

    split_totals = Counter()
    seen_paths = set()

    for record in records:
        split = record["split"]
        class_name = record["class_name"]
        image_path = record["image_path"]

        if split not in expected_counts:
            raise ValueError(f"Invalid split: {split}")

        if class_name not in class_to_idx:
            raise ValueError(f"Unknown class: {class_name}")

        if int(record["label"]) != class_to_idx[class_name]:
            raise ValueError(
                f"Incorrect label for {class_name}: {record['label']}"
            )

        if image_path in seen_paths:
            raise ValueError(f"Duplicate image path: {image_path}")

        seen_paths.add(image_path)

        split_totals[split] += 1
        class_split_counts[split][class_name] += 1

    for split, expected_per_class in expected_counts.items():
        for class_name, count in class_split_counts[split].items():
            if count != expected_per_class:
                raise ValueError(
                    f"{split}/{class_name}: expected "
                    f"{expected_per_class}, got {count}"
                )

        if len(class_split_counts[split]) != NUM_CLASSES:
            raise ValueError(
                f"{split}: expected {NUM_CLASSES} classes"
            )

    expected_totals = {
        "train": 68175,
        "validation": 7575,
        "test": 25250,
    }

    if dict(split_totals) != expected_totals:
        raise ValueError(
            f"Unexpected split totals: {dict(split_totals)}"
        )

    if len(records) != 101000:
        raise ValueError(
            f"Expected 101000 records, got {len(records)}"
        )


def save_manifest(
    records: list[dict],
    output_path: Path,
) -> None:
    """Write manifest to CSV."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=MANIFEST_COLUMNS,
        )

        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Food-101 split manifest"
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Path to extracted Food-101 directory",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV manifest path",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()

    print("Loading Food-101 metadata...")

    classes = load_classes(dataset_root)

    train_metadata = load_metadata(
        dataset_root / "meta" / "train.json",
        classes,
        OFFICIAL_TRAIN_PER_CLASS,
    )

    test_metadata = load_metadata(
        dataset_root / "meta" / "test.json",
        classes,
        OFFICIAL_TEST_PER_CLASS,
    )

    print("Building stratified manifest...")

    records = build_manifest(
        classes,
        train_metadata,
        test_metadata,
        seed=args.seed,
    )

    print("Validating manifest...")

    validate_manifest(records, classes)

    save_manifest(records, args.output)

    print("\n========== MANIFEST SUMMARY ==========")
    print(f"Seed: {args.seed}")
    print(f"Classes: {len(classes)}")
    print(f"Total images: {len(records)}")

    for split in ("train", "validation", "test"):
        count = sum(
            record["split"] == split
            for record in records
        )
        print(f"{split}: {count}")

    print(f"\nSaved manifest: {args.output}")
    print("MANIFEST VALIDATION: PASSED")


if __name__ == "__main__":
    main()
