"""Independently validate a Food-101 manifest against official metadata."""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path, PurePosixPath


EXPECTED_PER_CLASS = {"train": 675, "validation": 75, "test": 250}
EXPECTED_TOTALS = {"train": 68175, "validation": 7575, "test": 25250}
MANIFEST_COLUMNS = {"image_path", "class_name", "label", "split"}
NUM_CLASSES = 101
MAX_EXAMPLES = 5


class Check:
    """Collect and report a check's failures without flooding the output."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.count = 0
        self.examples: list[str] = []

    def fail(self, detail: str) -> None:
        """Record one failure and retain a bounded number of examples."""
        self.count += 1
        if len(self.examples) < MAX_EXAMPLES:
            self.examples.append(detail)

    def report(self) -> bool:
        """Print the result and return whether the check passed."""
        if not self.count:
            print(f"[PASS] {self.name}")
            return True
        print(f"[FAIL] {self.name}: {self.count} problem(s)")
        for example in self.examples:
            print(f"  - {example}")
        return False


def valid_image_path(path: str, class_name: str) -> bool:
    """Require a canonical, relative Food-101 image path for the class."""
    parts = path.split("/")
    return (
        len(parts) == 3
        and parts[0] == "images"
        and parts[1] == class_name
        and bool(parts[2])
        and parts[2] not in {".", ".."}
        and not parts[2].startswith(".")
        and parts[2].endswith(".jpg")
        and len(parts[2]) > len(".jpg")
        and "\\" not in path
        and all(part == part.strip() and part not in {"", ".", ".."} for part in parts)
        and not any(ord(char) < 32 or ord(char) == 127 for char in path)
        and PurePosixPath(path).as_posix() == path
    )


def load_classes(dataset_root: Path) -> list[str]:
    """Read the ordered class mapping; reject unusable or ambiguous metadata."""
    path = dataset_root / "meta" / "classes.txt"
    classes = path.read_text(encoding="utf-8").splitlines()
    if len(classes) != NUM_CLASSES or len(set(classes)) != NUM_CLASSES:
        raise ValueError(f"{path}: expected 101 distinct class names, got {len(classes)} lines and {len(set(classes))} distinct names")
    for name in classes:
        if not valid_image_path(f"images/{name}/sample.jpg", name):
            raise ValueError(f"{path}: invalid class name {name!r}")
    return classes


def unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Reject duplicate JSON object keys rather than silently discarding them."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def load_metadata(path: Path, classes: list[str], expected_count: int) -> set[str]:
    """Read and validate metadata, returning canonical relative image paths."""
    with path.open(encoding="utf-8") as file:
        metadata = json.load(file, object_pairs_hook=unique_json_object)
    if not isinstance(metadata, dict) or set(metadata) != set(classes):
        raise ValueError(f"{path}: class keys must exactly match classes.txt")
    images: set[str] = set()
    for class_name in classes:
        image_ids = metadata[class_name]
        if not isinstance(image_ids, list) or len(image_ids) != expected_count:
            raise ValueError(f"{path}: {class_name} must contain {expected_count} image IDs")
        for image_id in image_ids:
            if not isinstance(image_id, str):
                raise ValueError(f"{path}: non-string image ID in {class_name}")
            image_path = f"images/{image_id}"
            if not image_path.endswith(".jpg"):
                image_path += ".jpg"
            if not valid_image_path(image_path, class_name):
                raise ValueError(f"{path}: invalid image ID {image_id!r} for {class_name}")
            if image_path in images:
                raise ValueError(f"{path}: duplicate image ID {image_id!r}")
            images.add(image_path)
    return images


def validate_manifest(manifest: Path, classes: list[str], official_train: set[str], official_test: set[str]) -> bool:
    """Report all manifest checks and return whether every check passed."""
    checks = {
        key: Check(name)
        for key, name in (
            ("split", "Split names"),
            ("class", "Class names and coverage"),
            ("label", "Labels match classes.txt order"),
            ("path", "Relative image path format and class"),
            ("duplicate", "Duplicate image paths"),
            ("overlap", "Images in multiple splits"),
            ("class_counts", "Per-class split counts"),
            ("totals", "Split and overall totals"),
            ("train_membership", "Train + validation match official train.json"),
            ("test_membership", "Test matches official test.json"),
        )
    }
    class_labels = {name: index for index, name in enumerate(classes)}
    split_counts: Counter[str] = Counter()
    class_counts: Counter[tuple[str, str]] = Counter()
    seen: dict[str, str] = {}
    train_images: set[str] = set()
    test_images: set[str] = set()
    total = 0

    with manifest.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, strict=True)
        if reader.fieldnames is None or len(reader.fieldnames) != len(MANIFEST_COLUMNS) or set(reader.fieldnames) != MANIFEST_COLUMNS:
            raise ValueError(f"{manifest}: expected CSV columns {', '.join(sorted(MANIFEST_COLUMNS))} exactly once")
        for row in reader:
            line = reader.line_num
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{manifest}: malformed CSV row ending at line {line}")
            image_path = row["image_path"]
            class_name = row["class_name"]
            label = row["label"]
            split = row["split"]
            total += 1
            split_counts[split] += 1
            class_counts[(split, class_name)] += 1
            if split not in EXPECTED_PER_CLASS:
                checks["split"].fail(f"line {line}: {split!r}")
            if class_name not in class_labels:
                checks["class"].fail(f"line {line}: unknown class {class_name!r}")
            elif label != str(class_labels[class_name]):
                checks["label"].fail(f"line {line}: {class_name!r} has label {label!r}, expected {class_labels[class_name]}")
            if class_name not in class_labels and label:
                checks["label"].fail(f"line {line}: cannot map label {label!r} for unknown class {class_name!r}")
            if not valid_image_path(image_path, class_name):
                checks["path"].fail(f"line {line}: {image_path!r} for class {class_name!r}")
            if image_path in seen:
                checks["duplicate"].fail(f"line {line}: {image_path!r}")
                if seen[image_path] != split:
                    checks["overlap"].fail(f"line {line}: {image_path!r} in {seen[image_path]!r} and {split!r}")
            else:
                seen[image_path] = split
            if split in {"train", "validation"}:
                train_images.add(image_path)
            elif split == "test":
                test_images.add(image_path)

    for class_name in classes:
        if not any(class_counts[(split, class_name)] for split in EXPECTED_PER_CLASS):
            checks["class"].fail(f"missing class {class_name!r}")
        for split, expected in EXPECTED_PER_CLASS.items():
            actual = class_counts[(split, class_name)]
            if actual != expected:
                checks["class_counts"].fail(f"{split}/{class_name}: expected {expected}, got {actual}")
    for split, expected in EXPECTED_TOTALS.items():
        if split_counts[split] != expected:
            checks["totals"].fail(f"{split}: expected {expected}, got {split_counts[split]}")
    if total != 101000:
        checks["totals"].fail(f"overall: expected 101000, got {total}")
    for name, actual, expected in (
        ("train_membership", train_images, official_train),
        ("test_membership", test_images, official_test),
    ):
        for path in sorted(expected - actual):
            checks[name].fail(f"missing: {path}")
        for path in sorted(actual - expected):
            checks[name].fail(f"unexpected: {path}")
    print(f"Manifest: {total:,} rows; " + ", ".join(f"{split}: {split_counts[split]:,}" for split in EXPECTED_TOTALS))
    results = [check.report() for check in checks.values()]
    return all(results)


def main() -> int:
    """Parse CLI arguments, validate inputs and return a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True, help="Food-101 root containing meta/")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to the CSV manifest")
    args = parser.parse_args()
    try:
        classes = load_classes(args.dataset_root)
        print("[PASS] classes.txt: 101 distinct, valid classes")
        official_train = load_metadata(args.dataset_root / "meta" / "train.json", classes, 750)
        official_test = load_metadata(args.dataset_root / "meta" / "test.json", classes, 250)
        if official_train & official_test:
            raise ValueError("official train.json and test.json contain overlapping image IDs")
        print("[PASS] Official metadata: class keys, counts, paths, uniqueness and disjoint splits")
        passed = validate_manifest(args.manifest, classes, official_train, official_test)
    except (OSError, UnicodeError, ValueError, csv.Error) as exc:
        print(f"[FAIL] Input error: {exc}", file=sys.stderr)
        passed = False
    print("VALIDATION PASSED" if passed else "VALIDATION FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
