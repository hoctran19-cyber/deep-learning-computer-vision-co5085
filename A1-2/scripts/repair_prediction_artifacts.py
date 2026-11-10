"""Rebuild Food-101 prediction analysis artifacts from verified numeric labels.

This script repairs class-name mappings in an existing prediction export. It
does not load a model, train, or run inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Sequence


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PREDICTIONS = (
    PROJECT_DIR / "outputs" / "report_artifacts" / "metrics" / "prediction_results.csv"
)
DEFAULT_MANIFEST = PROJECT_DIR / "data" / "manifests" / "food101_split_seed42.csv"
DEFAULT_CLASSES = PROJECT_DIR / "data" / "food-101-metadata" / "meta" / "classes.txt"
DEFAULT_FINAL_METRICS = (
    PROJECT_DIR / "outputs" / "final_evaluation" / "test_metrics.json"
)
DEFAULT_ADDITIONAL_METRICS = (
    PROJECT_DIR / "outputs" / "report_artifacts" / "metrics" / "additional_metrics.json"
)
DEFAULT_OUTPUT = PROJECT_DIR / "outputs" / "repaired_artifacts"
EXPECTED_ROWS = 25_250
NUM_CLASSES = 101
METRIC_TOLERANCE = 1e-5
SAMPLE_SIZE = 3

LOGGER = logging.getLogger("repair_prediction_artifacts")

PREDICTION_COLUMNS = {
    "image_path": ("image_path", "path", "filepath", "file_path", "image"),
    "true_label": ("true_label", "label_true", "y_true", "target_label"),
    "pred_label": (
        "predicted_label",
        "pred_label",
        "prediction_label",
        "y_pred",
        "predicted_class_id",
    ),
    "true_class": ("true_class", "actual_class", "ground_truth_class"),
    "pred_class": ("predicted_class", "pred_class", "prediction_class"),
}
MANIFEST_COLUMNS = {
    "image_path": ("image_path", "path", "filepath", "file_path", "image"),
    "label": ("label", "class_id", "target", "class_idx"),
    "class_name": ("class_name", "class", "category"),
    "split": ("split", "subset", "set"),
}


class RepairError(ValueError):
    """Raised when source artifacts cannot be safely aligned or repaired."""


@dataclass(frozen=True)
class VerifiedPrediction:
    """One prediction row after manifest verification and class-name repair."""

    image_path: str
    normalized_path: str
    true_label: int
    pred_label: int
    true_class: str
    pred_class: str
    confidence: float | None
    original: dict[str, str]


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, strict=True)
        if reader.fieldnames is None:
            raise RepairError(f"{path}: missing CSV header")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise RepairError(f"{path}: duplicate CSV column names")
        rows = list(reader)
        if any(None in row or any(value is None for value in row.values()) for row in rows):
            raise RepairError(f"{path}: malformed CSV row")
        return list(reader.fieldnames), rows


def _resolve_columns(
    fieldnames: Sequence[str],
    candidates: dict[str, tuple[str, ...]],
    source: Path,
) -> dict[str, str]:
    by_lower = {name.casefold(): name for name in fieldnames}
    resolved: dict[str, str] = {}
    for purpose, aliases in candidates.items():
        matches = [by_lower[name.casefold()] for name in aliases if name.casefold() in by_lower]
        if len(matches) != 1:
            raise RepairError(
                f"{source}: expected exactly one column for {purpose}; "
                f"candidates={aliases}, matches={matches}, available={list(fieldnames)}"
            )
        resolved[purpose] = matches[0]
    return resolved


def _print_input_summary(
    path: Path,
    columns: Sequence[str],
    rows: Sequence[dict[str, str]],
) -> None:
    LOGGER.info("%s: columns=%s rows=%d", path, list(columns), len(rows))
    for index, row in enumerate(rows[:SAMPLE_SIZE], start=1):
        LOGGER.info("%s sample %d: %s", path.name, index, row)


def _load_classes(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    if len(lines) != NUM_CLASSES:
        raise RepairError(f"{path}: expected {NUM_CLASSES} class lines, found {len(lines)}")
    if any(not name.strip() or name != name.strip() for name in lines):
        raise RepairError(f"{path}: class names must be non-empty and have no surrounding whitespace")
    if len(set(lines)) != NUM_CLASSES:
        raise RepairError(f"{path}: class names must be unique")
    LOGGER.info(
        "%s: columns=['label', 'class_name'] rows=%d samples=%s",
        path,
        len(lines),
        [{"label": i, "class_name": name} for i, name in enumerate(lines[:SAMPLE_SIZE])],
    )
    return lines


def _normalize_image_path(value: str, source: Path, row_number: int) -> str:
    raw = value.strip().replace("\\", "/")
    if not raw:
        raise RepairError(f"{source}: row {row_number}: image path is empty")
    parts = PurePosixPath(raw).parts
    try:
        image_index = parts.index("images")
    except ValueError as error:
        raise RepairError(
            f"{source}: row {row_number}: path does not contain an images/ component: {value!r}"
        ) from error
    relative_parts = parts[image_index:]
    if (
        len(relative_parts) != 3
        or relative_parts[0] != "images"
        or any(part in {"", ".", ".."} for part in relative_parts)
    ):
        raise RepairError(
            f"{source}: row {row_number}: expected images/<class>/<filename>, got {value!r}"
        )
    return PurePosixPath(*relative_parts).as_posix()


def _parse_label(value: str, source: Path, row_number: int, column: str) -> int:
    try:
        label = int(value)
    except (TypeError, ValueError) as error:
        raise RepairError(
            f"{source}: row {row_number}: {column} must be an integer, got {value!r}"
        ) from error
    if not 0 <= label < NUM_CLASSES:
        raise RepairError(
            f"{source}: row {row_number}: {column}={label} is outside 0..{NUM_CLASSES - 1}"
        )
    return label


def _parse_confidence(
    row: dict[str, str],
    confidence_column: str | None,
    source: Path,
    row_number: int,
) -> float | None:
    if confidence_column is None or not row.get(confidence_column, "").strip():
        return None
    try:
        confidence = float(row[confidence_column])
    except ValueError as error:
        raise RepairError(
            f"{source}: row {row_number}: invalid confidence {row[confidence_column]!r}"
        ) from error
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise RepairError(
            f"{source}: row {row_number}: confidence must be finite and in [0, 1]"
        )
    return confidence


def _correction_counts(
    rows: Sequence[VerifiedPrediction],
    original_key: str,
    repaired_key: str,
) -> dict[str, list[dict[str, Any]]]:
    changes: Counter[tuple[str, str]] = Counter()
    for prediction in rows:
        old = prediction.original.get(original_key, "").strip() or "<empty>"
        new = getattr(prediction, repaired_key)
        if old != new:
            changes[(old, new)] += 1
    corrections: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (old, new), count in sorted(changes.items()):
        corrections[old].append({"correct_class_name": new, "count": count})
    return dict(corrections)


def verify_and_repair_rows(
    prediction_path: Path,
    manifest_path: Path,
    classes: Sequence[str],
    prediction_columns: Sequence[str],
    prediction_rows: Sequence[dict[str, str]],
    manifest_columns: Sequence[str],
    manifest_rows: Sequence[dict[str, str]],
) -> tuple[list[VerifiedPrediction], dict[str, Any]]:
    """Join predictions to test manifest paths and derive all names from labels."""
    prediction_map = _resolve_columns(
        prediction_columns, PREDICTION_COLUMNS, prediction_path
    )
    manifest_map = _resolve_columns(manifest_columns, MANIFEST_COLUMNS, manifest_path)
    LOGGER.info("Resolved prediction columns: %s", prediction_map)
    LOGGER.info("Resolved manifest columns: %s", manifest_map)

    by_path: dict[str, list[tuple[int, dict[str, str]]]] = defaultdict(list)
    duplicate_manifest_rows = 0
    for row_number, row in enumerate(manifest_rows, start=2):
        path = _normalize_image_path(row[manifest_map["image_path"]], manifest_path, row_number)
        by_path[path].append((row_number, row))
        if len(by_path[path]) > 1:
            duplicate_manifest_rows += 1

    test_manifest: dict[str, tuple[int, dict[str, str]]] = {}
    manifest_class_mapping_mismatches = 0
    for path, entries in by_path.items():
        test_entries = [
            (line, row)
            for line, row in entries
            if row[manifest_map["split"]].strip().casefold() == "test"
        ]
        if len(test_entries) > 1:
            continue
        if test_entries:
            line, row = test_entries[0]
            label = _parse_label(
                row[manifest_map["label"]], manifest_path, line, manifest_map["label"]
            )
            if row[manifest_map["class_name"]].strip() != classes[label]:
                manifest_class_mapping_mismatches += 1
            test_manifest[path] = (line, row)

    prediction_path_counts: Counter[str] = Counter()
    unmatched_predictions = 0
    duplicate_prediction_rows = 0
    label_mismatches = 0
    matched_rows = 0
    repaired: list[VerifiedPrediction] = []
    true_class_column = prediction_map["true_class"]
    pred_class_column = prediction_map["pred_class"]
    confidence_candidates = ("confidence", "probability", "score")
    confidence_columns = [
        column
        for column in prediction_columns
        if column.casefold() in {name.casefold() for name in confidence_candidates}
    ]
    if len(confidence_columns) > 1:
        raise RepairError(
            f"{prediction_path}: ambiguous confidence columns {confidence_columns}"
        )
    confidence_column = confidence_columns[0] if confidence_columns else None

    for row_number, row in enumerate(prediction_rows, start=2):
        path = _normalize_image_path(
            row[prediction_map["image_path"]], prediction_path, row_number
        )
        prediction_path_counts[path] += 1
        if prediction_path_counts[path] > 1:
            duplicate_prediction_rows += 1
        manifest_entry = test_manifest.get(path)
        if manifest_entry is None or len(by_path.get(path, [])) != 1:
            unmatched_predictions += 1
            continue

        manifest_line, manifest_row = manifest_entry
        true_label = _parse_label(
            row[prediction_map["true_label"]],
            prediction_path,
            row_number,
            prediction_map["true_label"],
        )
        pred_label = _parse_label(
            row[prediction_map["pred_label"]],
            prediction_path,
            row_number,
            prediction_map["pred_label"],
        )
        manifest_label = _parse_label(
            manifest_row[manifest_map["label"]],
            manifest_path,
            manifest_line,
            manifest_map["label"],
        )
        if true_label != manifest_label:
            label_mismatches += 1
            continue
        if manifest_row[manifest_map["class_name"]].strip() != classes[true_label]:
            continue

        repaired.append(
            VerifiedPrediction(
                image_path=row[prediction_map["image_path"]],
                normalized_path=path,
                true_label=true_label,
                pred_label=pred_label,
                true_class=classes[true_label],
                pred_class=classes[pred_label],
                confidence=_parse_confidence(
                    row, confidence_column, prediction_path, row_number
                ),
                original=row,
            )
        )
        matched_rows += 1

    prediction_keys = set(prediction_path_counts)
    test_keys = set(test_manifest)
    unmatched_manifest_rows = len(test_keys - prediction_keys)
    missing_or_duplicate_manifest = sum(
        1 for path in prediction_keys if len(by_path.get(path, [])) != 1
    )
    audit = {
        "input_columns": {
            "prediction": list(prediction_columns),
            "manifest": list(manifest_columns),
        },
        "input_rows": {
            "prediction": len(prediction_rows),
            "manifest": len(manifest_rows),
            "manifest_test": len(test_manifest),
        },
        "resolved_columns": {
            "prediction": prediction_map,
            "manifest": manifest_map,
            "confidence": confidence_column,
        },
        "alignment": {
            "expected_prediction_rows": EXPECTED_ROWS,
            "matched_prediction_rows": matched_rows,
            "unmatched_prediction_rows": unmatched_predictions,
            "unmatched_manifest_test_rows": unmatched_manifest_rows,
            "duplicate_prediction_rows": duplicate_prediction_rows,
            "duplicate_manifest_rows": duplicate_manifest_rows,
            "paths_with_non_unique_manifest_rows": missing_or_duplicate_manifest,
            "true_label_mismatches": label_mismatches,
            "manifest_class_mapping_mismatches": manifest_class_mapping_mismatches,
        },
        "class_mapping_corrections": {
            "true_class": _correction_counts(repaired, true_class_column, "true_class"),
            "pred_class": _correction_counts(repaired, pred_class_column, "pred_class"),
        },
    }

    if len(prediction_rows) != EXPECTED_ROWS:
        raise RepairError(
            f"Expected {EXPECTED_ROWS} prediction rows, found {len(prediction_rows)}"
        )
    if len(test_manifest) != EXPECTED_ROWS:
        raise RepairError(
            f"Expected {EXPECTED_ROWS} unique test manifest rows, found {len(test_manifest)}"
        )
    if (
        matched_rows != EXPECTED_ROWS
        or unmatched_predictions
        or unmatched_manifest_rows
        or duplicate_prediction_rows
        or duplicate_manifest_rows
        or missing_or_duplicate_manifest
        or label_mismatches
        or manifest_class_mapping_mismatches
        or len(repaired) != EXPECTED_ROWS
    ):
        LOGGER.error(
            "Alignment verification failed: %s",
            json.dumps(audit["alignment"], ensure_ascii=False, sort_keys=True),
        )
        raise RepairError(
            "Prediction-to-manifest verification failed; alignment counts are logged above"
        )
    return repaired, audit


def _load_reference_metrics(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    metrics = data.get("test_metrics", data)
    if not isinstance(metrics, dict):
        raise RepairError(f"{path}: expected a metrics JSON object")
    LOGGER.info(
        "%s: metric columns=%s rows=1 sample=%s",
        path,
        list(metrics),
        metrics,
    )
    result: dict[str, float] = {}
    for key in ("accuracy", "macro_f1"):
        value = metrics.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RepairError(f"{path}: missing numeric metric {key!r}")
        result[key] = float(value)
    return result


def _build_metrics(
    predictions: Sequence[VerifiedPrediction],
    classes: Sequence[str],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[list[int]], list[dict[str, Any]]]:
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        precision_recall_fscore_support,
    )

    y_true = np.asarray([item.true_label for item in predictions], dtype=np.int64)
    y_pred = np.asarray([item.pred_label for item in predictions], dtype=np.int64)
    labels = list(range(NUM_CLASSES))
    accuracy = float(accuracy_score(y_true, y_pred))
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=list(classes),
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    report_rows: list[dict[str, Any]] = []
    for class_name in classes:
        report_rows.append({"": class_name, **report[class_name]})
    report_rows.extend(
        [
            {
                "": "accuracy",
                "precision": accuracy,
                "recall": accuracy,
                "f1-score": accuracy,
                "support": accuracy,
            },
            {"": "macro avg", **report["macro avg"]},
            {"": "weighted avg", **report["weighted avg"]},
        ]
    )
    top_confusions: list[dict[str, Any]] = []
    for true_index in labels:
        for pred_index in labels:
            count = int(matrix[true_index, pred_index])
            if true_index != pred_index and count:
                top_confusions.append(
                    {
                        "True_Class": classes[true_index],
                        "Predicted_Class": classes[pred_index],
                        "Count": count,
                        "_true_index": true_index,
                        "_pred_index": pred_index,
                    }
                )
    top_confusions.sort(
        key=lambda row: (
            -row["Count"],
            row["_true_index"],
            row["_pred_index"],
        )
    )
    top_confusions = [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in top_confusions[:10]
    ]
    metrics = {
        "model": "resnet50_full_finetune",
        "num_test_samples": len(predictions),
        "num_classes": NUM_CLASSES,
        "accuracy": accuracy,
        "top1_accuracy": accuracy,
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "correct_predictions": int(np.count_nonzero(y_true == y_pred)),
        "misclassified_predictions": int(np.count_nonzero(y_true != y_pred)),
    }
    return metrics, report_rows, matrix.tolist(), top_confusions


def _write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[dict[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_confusion_matrix(matrix: Sequence[Sequence[int]], classes: Sequence[str], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    values = np.asarray(matrix, dtype=np.float64)
    row_sums = values.sum(axis=1, keepdims=True)
    normalized = np.divide(values, row_sums, out=np.zeros_like(values), where=row_sums != 0)
    figure, axis = plt.subplots(figsize=(18, 15))
    image = axis.imshow(normalized, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    figure.colorbar(image, ax=axis, label="Normalized frequency")
    labels = [name.replace("_", " ") for name in classes]
    ticks = range(len(classes))
    axis.set(
        title="ResNet50 Full Fine-tuning - Normalized Confusion Matrix",
        xlabel="Predicted Class",
        ylabel="True Class",
    )
    axis.set_xticks(ticks, labels, rotation=90, fontsize=5)
    axis.set_yticks(ticks, labels, fontsize=5)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_class_bars(
    rows: Sequence[dict[str, Any]],
    metric: str,
    title: str,
    xlabel: str,
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = list(reversed(rows))
    labels = [str(row[""]).replace("_", " ") for row in ordered]
    values = [float(row[metric]) for row in ordered]
    figure, axis = plt.subplots(figsize=(12, 8))
    bars = axis.barh(labels, values, color="#bd3b3b" if metric == "f1-score" else "#ce6c2c")
    axis.bar_label(bars, labels=[f"{value:.4f}" for value in values], padding=4)
    axis.set(title=title, xlabel=xlabel, ylabel="Class Name", xlim=(0, 1))
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_confused_pairs(
    pairs: Sequence[dict[str, Any]],
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = list(reversed(pairs))
    labels = [
        f"{row['True_Class'].replace('_', ' ')} → {row['Predicted_Class'].replace('_', ' ')}"
        for row in ordered
    ]
    values = [int(row["Count"]) for row in ordered]
    figure, axis = plt.subplots(figsize=(12, 8))
    bars = axis.barh(labels, values, color="#477f78")
    axis.bar_label(bars, padding=4)
    axis.set(title="Top 10 Most Confused Food Class Pairs", xlabel="Misclassified Test Images")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _resolve_dataset_image(dataset_root: Path, relative_path: str) -> Path:
    root = dataset_root.resolve()
    image = (root / relative_path).resolve()
    try:
        image.relative_to(root)
    except ValueError as error:
        raise RepairError(f"Image path escapes dataset root: {relative_path}") from error
    if not image.is_file():
        raise FileNotFoundError(f"Dataset image not found: {image}")
    return image


def _plot_prediction_examples(
    predictions: Sequence[VerifiedPrediction],
    dataset_root: Path,
    path: Path,
) -> dict[str, int]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    correct = [item for item in predictions if item.true_label == item.pred_label]
    incorrect = [item for item in predictions if item.true_label != item.pred_label]

    def rank(item: VerifiedPrediction) -> tuple[float, str]:
        return (item.confidence if item.confidence is not None else -1.0, item.normalized_path)

    selected_correct = sorted(correct, key=rank, reverse=True)[:12]
    selected_incorrect = sorted(incorrect, key=rank, reverse=True)[:12]
    selected = [(item, True) for item in selected_correct] + [
        (item, False) for item in selected_incorrect
    ]
    figure, axes = plt.subplots(6, 4, figsize=(16, 20))
    for axis, (item, is_correct) in zip(axes.flat, selected):
        image_path = _resolve_dataset_image(dataset_root, item.normalized_path)
        with Image.open(image_path) as source:
            axis.imshow(source.convert("RGB"))
        status = "CORRECT" if is_correct else "MISCLASSIFIED"
        confidence = (
            f" ({item.confidence:.2%})" if item.confidence is not None else ""
        )
        axis.set_title(
            f"{status}\nGT: {item.true_class}\nPred: {item.pred_class}{confidence}",
            fontsize=8,
        )
        axis.axis("off")
    for axis in list(axes.flat)[len(selected):]:
        axis.axis("off")
    figure.suptitle(
        f"ResNet50 Full Fine-tuning - Test Prediction Examples\n"
        f"{len(selected_correct)} Correct | {len(selected_incorrect)} Misclassified",
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return {"correct_examples": len(selected_correct), "misclassified_examples": len(selected_incorrect)}


def _compare_metrics(
    rebuilt: dict[str, Any],
    final_reference: dict[str, float],
    additional_reference: dict[str, float] | None,
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    passed = True
    for source_name, source in (
        ("final_test_metrics", final_reference),
        ("additional_metrics", additional_reference),
    ):
        if source is None:
            continue
        comparisons[source_name] = {}
        for key in ("accuracy", "macro_f1"):
            delta = float(rebuilt[key]) - source[key]
            within_tolerance = abs(delta) <= METRIC_TOLERANCE
            comparisons[source_name][key] = {
                "rebuilt": float(rebuilt[key]),
                "reference": source[key],
                "difference": delta,
                "tolerance": METRIC_TOLERANCE,
                "status": "PASS" if within_tolerance else "FAIL",
            }
            passed = passed and within_tolerance
    note = (
        "Rebuilt aggregate metrics are within tolerance of each existing reference. "
        "Reference files are preserved unchanged; small differences are recorded "
        "rather than replacing source results."
        if passed
        else "At least one rebuilt aggregate metric exceeds the comparison tolerance. "
        "No reference value was changed; inspect the per-source differences before use."
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "tolerance": METRIC_TOLERANCE,
        "explanation": note,
        "comparisons": comparisons,
    }


def repair(args: argparse.Namespace) -> dict[str, Any]:
    prediction_columns, prediction_rows = _read_csv(args.prediction_csv)
    manifest_columns, manifest_rows = _read_csv(args.manifest_csv)
    _print_input_summary(args.prediction_csv, prediction_columns, prediction_rows)
    _print_input_summary(args.manifest_csv, manifest_columns, manifest_rows)
    classes = _load_classes(args.classes_file)
    predictions, audit = verify_and_repair_rows(
        args.prediction_csv,
        args.manifest_csv,
        classes,
        prediction_columns,
        prediction_rows,
        manifest_columns,
        manifest_rows,
    )
    audit["class_mapping"] = {
        "classes_file": str(args.classes_file),
        "num_classes": len(classes),
        "label_to_class": {str(index): name for index, name in enumerate(classes)},
    }
    LOGGER.info(
        "Alignment: matched=%d unmatched_predictions=%d unmatched_manifest_test=%d "
        "duplicate_predictions=%d duplicate_manifest_rows=%d true_label_mismatches=%d",
        audit["alignment"]["matched_prediction_rows"],
        audit["alignment"]["unmatched_prediction_rows"],
        audit["alignment"]["unmatched_manifest_test_rows"],
        audit["alignment"]["duplicate_prediction_rows"],
        audit["alignment"]["duplicate_manifest_rows"],
        audit["alignment"]["true_label_mismatches"],
    )
    LOGGER.info(
        "Class-name corrections: true_class=%s pred_class=%s",
        audit["class_mapping_corrections"]["true_class"],
        audit["class_mapping_corrections"]["pred_class"],
    )

    if not args.dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {args.dataset_root}")
    if not (args.dataset_root / "images").is_dir():
        raise FileNotFoundError(
            f"Dataset images directory not found: {args.dataset_root / 'images'}"
        )
    final_reference = _load_reference_metrics(args.final_test_metrics)
    additional_reference = (
        _load_reference_metrics(args.additional_metrics)
        if args.additional_metrics is not None
        else None
    )
    metrics, report_rows, matrix, pairs = _build_metrics(predictions, classes)
    correct_count = metrics["correct_predictions"]
    incorrect_count = metrics["misclassified_predictions"]
    matrix_total = sum(sum(row) for row in matrix)
    support_by_label = Counter(item.true_label for item in predictions)
    expected_support_per_class = EXPECTED_ROWS // NUM_CLASSES
    class_support = [
        {
            "label": label,
            "class_name": classes[label],
            "support": support_by_label[label],
            "expected_support": expected_support_per_class,
            "status": (
                "PASS"
                if support_by_label[label] == expected_support_per_class
                else "FAIL"
            ),
        }
        for label in range(NUM_CLASSES)
    ]
    support_status = (
        "PASS" if all(item["status"] == "PASS" for item in class_support) else "FAIL"
    )
    validation_results = {
        "prediction_count": len(predictions),
        "expected_prediction_count": EXPECTED_ROWS,
        "all_predictions_match_one_test_manifest_row": len(predictions) == EXPECTED_ROWS,
        "all_true_labels_match_manifest": audit["alignment"]["true_label_mismatches"] == 0,
        "test_class_support": {
            "expected_support_per_class": expected_support_per_class,
            "status": support_status,
            "classes": class_support,
        },
        "confusion_matrix_total": matrix_total,
        "correct_plus_incorrect": correct_count + incorrect_count,
        "correct_predictions": correct_count,
        "incorrect_predictions": incorrect_count,
        "confusion_matrix_total_valid": matrix_total == EXPECTED_ROWS,
        "correct_plus_incorrect_valid": correct_count + incorrect_count == EXPECTED_ROWS,
        "metrics_match_final_test_within_tolerance": None,
    }
    comparison = _compare_metrics(metrics, final_reference, additional_reference)
    validation_results["metrics_match_final_test_within_tolerance"] = (
        comparison["status"] == "PASS"
    )
    validation_results["aggregate_metrics_comparison"] = comparison
    audit["metrics"] = metrics
    audit["validation_results"] = validation_results
    audit["sources"] = {
        "prediction_csv": str(args.prediction_csv),
        "manifest_csv": str(args.manifest_csv),
        "classes_file": str(args.classes_file),
        "final_test_metrics": str(args.final_test_metrics),
        "additional_metrics": (
            str(args.additional_metrics) if args.additional_metrics is not None else None
        ),
    }

    # Do not publish a result set if core aggregate invariants fail.
    core_checks = (
        validation_results["prediction_count"] == EXPECTED_ROWS
        and validation_results["all_predictions_match_one_test_manifest_row"]
        and validation_results["all_true_labels_match_manifest"]
        and validation_results["test_class_support"]["status"] == "PASS"
        and validation_results["confusion_matrix_total_valid"]
        and validation_results["correct_plus_incorrect_valid"]
    )
    if not core_checks:
        raise RepairError("Post-rebuild core validation failed")

    output_dir: Path = args.output_dir
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists; refusing to overwrite: {output_dir}"
        )
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        audit["output_directory"] = str(output_dir)
        audit["validation_results"]["status"] = (
            "PASS" if comparison["status"] == "PASS" else "FAIL"
        )
        _write_repaired_artifacts(
            staging_dir,
            predictions,
            classes,
            report_rows,
            matrix,
            pairs,
            metrics,
            audit,
            args.dataset_root,
        )
        (staging_dir / "mapping_audit.json").write_text(
            json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.rename(staging_dir, output_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    return audit


def _write_repaired_artifacts(
    output_dir: Path,
    predictions: Sequence[VerifiedPrediction],
    classes: Sequence[str],
    report_rows: Sequence[dict[str, Any]],
    matrix: Sequence[Sequence[int]],
    pairs: Sequence[dict[str, Any]],
    metrics: dict[str, Any],
    audit: dict[str, Any],
    dataset_root: Path,
) -> None:
    corrected_rows = [
        {
            "image_path": item.image_path,
            "true_label": item.true_label,
            "true_class": item.true_class,
            "predicted_label": item.pred_label,
            "predicted_class": item.pred_class,
            "confidence": "" if item.confidence is None else item.confidence,
            "correct": item.true_label == item.pred_label,
        }
        for item in predictions
    ]
    _write_csv(
        output_dir / "prediction_results.csv",
        (
            "image_path",
            "true_label",
            "true_class",
            "predicted_label",
            "predicted_class",
            "confidence",
            "correct",
        ),
        corrected_rows,
    )
    _write_csv(
        output_dir / "classification_report.csv",
        ("", "precision", "recall", "f1-score", "support"),
        report_rows,
    )
    matrix_rows = [
        {"": classes[i], **{classes[j]: matrix[i][j] for j in range(NUM_CLASSES)}}
        for i in range(NUM_CLASSES)
    ]
    _write_csv(output_dir / "confusion_matrix.csv", ("", *classes), matrix_rows)

    per_class = [
        row for row in report_rows if row[""] in classes
    ]
    worst_f1 = sorted(
        per_class,
        key=lambda row: (row["f1-score"], classes.index(row[""])),
    )[:10]
    worst_recall = sorted(
        per_class,
        key=lambda row: (row["recall"], classes.index(row[""])),
    )[:10]
    _write_csv(
        output_dir / "per_class_error_analysis.csv",
        ("", "precision", "recall", "f1-score", "support"),
        per_class,
    )
    _write_csv(
        output_dir / "worst10_classes_by_f1.csv",
        ("", "precision", "recall", "f1-score", "support"),
        worst_f1,
    )
    _write_csv(
        output_dir / "worst10_classes_by_recall.csv",
        ("", "precision", "recall", "f1-score", "support"),
        worst_recall,
    )
    _write_csv(
        output_dir / "top10_confused_class_pairs.csv",
        ("True_Class", "Predicted_Class", "Count"),
        pairs,
    )
    (output_dir / "additional_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    summary = {
        "total_classes": NUM_CLASSES,
        "alignment_verification_status": "Verified by normalized image_path and numeric true_label",
        "limitations": (
            "Legacy class-name fields were not used to compute metrics; names were rebuilt "
            "from numeric labels and classes.txt."
        ),
        "worst_10_classes_by_f1": {
            row[""]: float(row["f1-score"]) for row in worst_f1
        },
        "worst_10_classes_by_recall": {
            row[""]: float(row["recall"]) for row in worst_recall
        },
        "top_10_confusion_pairs": [
            {
                "true_class": row["True_Class"],
                "predicted_class": row["Predicted_Class"],
                "count": int(row["Count"]),
            }
            for row in pairs
        ],
    }
    (output_dir / "error_analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    _plot_confusion_matrix(
        matrix, classes, output_dir / "confusion_matrix.png"
    )
    _plot_class_bars(
        worst_f1,
        "f1-score",
        "Top 10 Food-101 Classes with Lowest F1-score (ResNet50)",
        "F1-score",
        output_dir / "worst10_classes_by_f1.png",
    )
    _plot_class_bars(
        worst_recall,
        "recall",
        "Top 10 Food-101 Classes with Lowest Recall (ResNet50)",
        "Recall",
        output_dir / "worst10_classes_by_recall.png",
    )
    _plot_confused_pairs(pairs, output_dir / "top10_confused_class_pairs.png")
    example_counts = _plot_prediction_examples(
        predictions, dataset_root, output_dir / "prediction_examples.png"
    )
    audit["prediction_examples"] = example_counts


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-csv", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--classes-file", type=Path, default=DEFAULT_CLASSES)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Food-101 root containing images/ for rebuilding prediction_examples.png",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--final-test-metrics", type=Path, default=DEFAULT_FINAL_METRICS)
    parser.add_argument(
        "--additional-metrics",
        type=Path,
        default=DEFAULT_ADDITIONAL_METRICS,
        help="Optional second aggregate reference, defaults to existing additional_metrics.json",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        stream=sys.stdout,
    )
    try:
        args = _parse_args(argv)
        audit = repair(args)
        result = audit["validation_results"]
        status = result["status"]
        LOGGER.info(
            "%s: predictions=%d correct=%d incorrect=%d accuracy=%.8f macro_f1=%.8f",
            status,
            result["prediction_count"],
            result["correct_predictions"],
            result["incorrect_predictions"],
            audit["metrics"]["accuracy"],
            audit["metrics"]["macro_f1"],
        )
        LOGGER.info("Repaired artifacts: %s", audit["output_directory"])
        if status != "PASS":
            LOGGER.error(
                "Aggregate metrics differ from a reference by more than %.1e; "
                "see mapping_audit.json for deltas.",
                METRIC_TOLERANCE,
            )
            return 1
        LOGGER.info("SUMMARY: PASS")
        return 0
    except Exception as error:
        LOGGER.exception("SUMMARY: FAIL — %s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
