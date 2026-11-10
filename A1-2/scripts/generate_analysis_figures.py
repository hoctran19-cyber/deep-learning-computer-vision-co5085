"""Generate Food-101 experiment and selected-model analysis figures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "outputs" / "figures"
MANIFEST_PATH = PROJECT_DIR / "data" / "manifests" / "food101_split_seed42.csv"
EXPERIMENT_LABELS = {
    "resnet50_frozen": "ResNet50 Frozen",
    "resnet50_partial": "ResNet50 Partial",
    "resnet50_full_finetune": "ResNet50 Full",
    "vit_b16_scratch": "ViT-B/16 Scratch",
    "vit_b16_frozen": "ViT-B/16 Frozen",
    "vit_b16_partial": "ViT-B/16 Partial",
    "vit_b16_full_finetune": "ViT-B/16 Full",
}
SELECTED_EXPERIMENT = "resnet50_full_finetune"
METRIC_KEYS = {
    "validation_accuracy": "validation_accuracy",
    "validation_macro_f1": "validation_macro_f1",
    "training_time": "fit_wall_seconds_including_validation_and_io",
}


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _read_history(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def _load_experiments() -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    summaries = {}
    histories = {}
    experiments_dir = PROJECT_DIR / "outputs" / "experiments"
    for name in EXPERIMENT_LABELS:
        directory = experiments_dir / name
        summary_path = directory / "summary.json"
        history_path = directory / "training_log.jsonl"
        if summary_path.is_file():
            summaries[name] = _read_json(summary_path)
        else:
            print(f"Skipping summary metrics for {name}: missing {summary_path}")
        if history_path.is_file():
            histories[name] = _read_history(history_path)
        else:
            print(f"Skipping training curves for {name}: missing {history_path}")
    return summaries, histories


def _save_figure(figure, path: Path) -> None:
    figure.savefig(path, dpi=180, bbox_inches="tight")
    _pyplot().close(figure)
    print(f"Created {path}")


def _plot_training_curves(histories: dict[str, list[dict[str, Any]]]) -> None:
    plt = _pyplot()
    colors = plt.get_cmap("tab10")
    for metric, title, ylabel, filename in (
        ("loss", "Training and validation loss", "Loss", "training_loss.png"),
        (
            "accuracy",
            "Training and validation accuracy",
            "Accuracy",
            "training_accuracy.png",
        ),
    ):
        figure, axis = plt.subplots(figsize=(13, 7))
        for index, (name, rows) in enumerate(histories.items()):
            color = colors(index % 10)
            for phase, linestyle in (("train", "-"), ("validation", "--")):
                epochs = []
                values = []
                for row in rows:
                    value = row.get(phase, {}).get(metric)
                    if value is not None:
                        epochs.append(row["epoch"])
                        values.append(value)
                if values:
                    axis.plot(
                        epochs,
                        values,
                        color=color,
                        linestyle=linestyle,
                        linewidth=1.8,
                        label=f"{EXPERIMENT_LABELS[name]} {phase}",
                    )
        axis.set(title=title, xlabel="Epoch", ylabel=ylabel)
        if metric == "accuracy":
            axis.set_ylim(0, 1)
        axis.grid(True, alpha=0.25)
        axis.legend(ncol=2, fontsize=8, frameon=False)
        _save_figure(figure, OUTPUT_DIR / filename)

    figure, axis = plt.subplots(figsize=(13, 7))
    for index, (name, rows) in enumerate(histories.items()):
        epochs = []
        values = []
        for row in rows:
            value = row.get("validation", {}).get("macro_f1")
            if value is not None:
                epochs.append(row["epoch"])
                values.append(value)
        if values:
            axis.plot(
                epochs,
                values,
                color=colors(index % 10),
                linewidth=1.8,
                marker=".",
                markersize=4,
                label=EXPERIMENT_LABELS[name],
            )
    axis.set(
        title="Validation Macro-F1 by epoch",
        xlabel="Epoch",
        ylabel="Macro-F1",
        ylim=(0, 1),
    )
    axis.grid(True, alpha=0.25)
    axis.legend(ncol=2, fontsize=9, frameon=False)
    _save_figure(figure, OUTPUT_DIR / "validation_macro_f1_curves.png")


def _numeric_metric(summary: dict[str, Any], key: str) -> float | None:
    value = summary.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _plot_comparison(
    summaries: dict[str, dict[str, Any]],
    metric_key: str,
    title: str,
    ylabel: str,
    filename: str,
    scale: float = 1.0,
    annotation_format: str = ".2f",
) -> None:
    plt = _pyplot()
    names = []
    values = []
    for name in EXPERIMENT_LABELS:
        summary = summaries.get(name)
        value = _numeric_metric(summary, metric_key) if summary else None
        if value is not None:
            names.append(name)
            values.append(value * scale)
        else:
            print(f"Skipping {metric_key} for {name}: metric is unavailable")
    if not values:
        print(f"Skipping {filename}: no {metric_key} values are available")
        return

    figure, axis = plt.subplots(figsize=(11, 6))
    bars = axis.bar(
        [EXPERIMENT_LABELS[name] for name in names],
        values,
        color=[plt.get_cmap("tab10")(index % 10) for index in range(len(names))],
    )
    axis.set(title=title, ylabel=ylabel)
    axis.tick_params(axis="x", labelrotation=25)
    for tick in axis.get_xticklabels():
        tick.set_horizontalalignment("right")
    axis.grid(axis="y", alpha=0.25)
    if scale == 100:
        axis.set_ylim(0, 100)
        labels = [f"{value:.1f}%" for value in values]
    else:
        labels = [format(value, annotation_format) for value in values]
    axis.bar_label(bars, labels=labels, padding=3, fontsize=8)
    _save_figure(figure, OUTPUT_DIR / filename)


def _plot_comparisons(summaries: dict[str, dict[str, Any]]) -> None:
    _plot_comparison(
        summaries,
        METRIC_KEYS["validation_accuracy"],
        "Best validation accuracy by experiment",
        "Validation accuracy",
        "validation_accuracy_comparison.png",
        scale=100,
    )
    _plot_comparison(
        summaries,
        METRIC_KEYS["validation_macro_f1"],
        "Best validation Macro-F1 by experiment",
        "Validation Macro-F1",
        "validation_macro_f1_comparison.png",
        scale=100,
    )
    _plot_comparison(
        summaries,
        METRIC_KEYS["training_time"],
        "Experiment wall time (includes validation and I/O)",
        "Wall time (minutes)",
        "training_time_comparison.png",
        scale=1 / 60,
        annotation_format=".1f",
    )


def _load_classes(dataset_root: Path) -> list[str]:
    classes_path = dataset_root / "meta" / "classes.txt"
    if not classes_path.is_file():
        raise FileNotFoundError(f"Food-101 class list not found: {classes_path}")
    classes = [
        line.strip()
        for line in classes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(classes) != 101 or len(set(classes)) != 101:
        raise ValueError(f"Expected 101 unique Food-101 classes in {classes_path}")
    return classes


def _read_manifest(dataset_root: Path, split: str) -> list[dict[str, Any]]:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Food-101 manifest not found: {MANIFEST_PATH}")
    classes = _load_classes(dataset_root)
    root = dataset_root.resolve()
    samples = []
    with MANIFEST_PATH.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required = {"image_path", "class_name", "label", "split"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Manifest must contain columns: {sorted(required)}")
        for line_number, row in enumerate(reader, start=2):
            if row["split"] != split:
                continue
            label = int(row["label"])
            if not 0 <= label < len(classes) or classes[label] != row["class_name"]:
                raise ValueError(f"{MANIFEST_PATH}:{line_number}: class/label mismatch")
            image_path = (root / row["image_path"]).resolve()
            if not image_path.is_relative_to(root):
                raise ValueError(f"Image path escapes dataset root: {row['image_path']}")
            if not image_path.is_file():
                raise FileNotFoundError(f"Food-101 image not found: {image_path}")
            samples.append(
                {
                    "image_path": image_path,
                    "class_name": row["class_name"],
                    "label": label,
                }
            )
    return samples


def _plot_sample_grid(dataset_root: Path) -> None:
    from PIL import Image

    classes = _load_classes(dataset_root)
    samples = _read_manifest(dataset_root, "train")
    by_label: dict[int, list[dict[str, Any]]] = {}
    for sample in samples:
        by_label.setdefault(sample["label"], []).append(sample)
    selected_labels = [round(index * 100 / 15) for index in range(16)]
    randomizer = random.Random(42)
    selected = [
        randomizer.choice(by_label[label])
        for label in selected_labels
        if by_label.get(label)
    ]
    if len(selected) != 16:
        raise ValueError("Could not select 16 labeled images from the training split")

    plt = _pyplot()
    figure, axes = plt.subplots(4, 4, figsize=(14, 12))
    for axis, sample in zip(axes.flat, selected):
        with Image.open(sample["image_path"]) as image:
            axis.imshow(image.convert("RGB"))
        axis.set_title(classes[sample["label"]].replace("_", " "), fontsize=10)
        axis.axis("off")
    figure.suptitle("Food-101 training samples (labels from the manifest)", fontsize=16)
    _save_figure(figure, OUTPUT_DIR / "food101_sample_grid.png")


def _select_device(torch):
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _run_selected_model_inference(
    dataset_root: Path,
    checkpoint_path: Path,
    batch_size: int,
    num_workers: int,
) -> None:
    import numpy as np
    import torch
    from PIL import Image

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Selected-model checkpoint not found: {checkpoint_path}")
    final_evaluation = _read_json(
        PROJECT_DIR / "outputs" / "final_evaluation" / "test_metrics.json"
    )
    selected_name = final_evaluation.get("selected_experiment")
    if selected_name != SELECTED_EXPERIMENT:
        raise ValueError(
            f"Expected selected model {SELECTED_EXPERIMENT}, found {selected_name!r}"
        )
    summary = _read_json(
        PROJECT_DIR / "outputs" / "experiments" / selected_name / "summary.json"
    )
    expected_hash = summary.get("best_checkpoint_sha256")
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    actual_hash = digest.hexdigest()
    if expected_hash and actual_hash != expected_hash:
        raise ValueError(
            "Checkpoint SHA-256 does not match the selected experiment summary"
        )

    sys_path = str(PROJECT_DIR)
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    from src.dataset import Food101Dataset, create_dataloader
    from src.models import build_model
    from src.transforms import build_eval_transform

    import yaml

    with (PROJECT_DIR / "configs" / "config.yaml").open(encoding="utf-8") as file:
        training_config = yaml.safe_load(file)
    data_config = training_config["data"]
    dataset = Food101Dataset(
        dataset_root,
        MANIFEST_PATH,
        "test",
        build_eval_transform(
            resize_short_edge=data_config["resize_short_edge"],
            crop_size=data_config["image_size"],
        ),
    )
    if len(dataset) != 25250:
        raise ValueError(f"Expected 25250 test images, found {len(dataset)}")
    loader = create_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    device = _select_device(torch)
    model = build_model("resnet50", num_classes=101, pretrained=False).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint does not contain model_state_dict")
    if checkpoint.get("epoch") != summary.get("best_epoch"):
        raise ValueError("Checkpoint epoch does not match the recorded best epoch")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    classes = dataset.classes
    confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)
    correct_examples = []
    incorrect_examples = []
    sample_offset = 0
    with torch.inference_mode():
        for images, labels in loader:
            predictions = model(images.to(device)).argmax(dim=1).cpu().numpy()
            ground_truth = labels.numpy()
            np.add.at(confusion, (ground_truth, predictions), 1)
            for offset, (truth, prediction) in enumerate(
                zip(ground_truth, predictions)
            ):
                sample = dataset.samples[sample_offset + offset]
                example = (
                    sample["image_path"],
                    classes[int(truth)],
                    classes[int(prediction)],
                )
                if truth == prediction:
                    correct_examples.append(example)
                else:
                    incorrect_examples.append(example)
            sample_offset += len(ground_truth)

    _plot_confusion_matrix(confusion, classes)
    _plot_prediction_examples(correct_examples, incorrect_examples)
    print(
        "Checkpoint inference completed for the confusion matrix and examples only; "
        "aggregate test metrics were not calculated or saved."
    )


def _plot_confusion_matrix(confusion, classes: list[str]) -> None:
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(19, 17))
    image = axis.imshow(confusion, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis, label="Image count", fraction=0.035, pad=0.02)
    ticks = range(len(classes))
    labels = [name.replace("_", " ") for name in classes]
    axis.set(
        title="Test split confusion matrix - ResNet50 full fine-tuning",
        xlabel="Predicted label",
        ylabel="Ground truth",
    )
    axis.set_xticks(ticks, labels, rotation=90, fontsize=5)
    axis.set_yticks(ticks, labels, fontsize=5)
    axis.tick_params(axis="both", length=0)
    _save_figure(
        figure, OUTPUT_DIR / "resnet50_full_finetune_confusion_matrix.png"
    )


def _plot_prediction_examples(correct_examples, incorrect_examples) -> None:
    from PIL import Image

    randomizer = random.Random(42)
    randomizer.shuffle(correct_examples)
    randomizer.shuffle(incorrect_examples)
    selected = [
        (example, True) for example in correct_examples[:8]
    ] + [
        (example, False) for example in incorrect_examples[:8]
    ]
    if not selected:
        print("Skipping prediction examples: inference produced no samples")
        return

    plt = _pyplot()
    columns = 4
    rows = (len(selected) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(14, 3.7 * rows), squeeze=False)
    for axis, (example, correct) in zip(axes.flat, selected):
        image_path, ground_truth, prediction = example
        with Image.open(image_path) as image:
            axis.imshow(image.convert("RGB"))
        axis.set_title(
            f"{'Correct' if correct else 'Misclassified'}\n"
            f"GT: {ground_truth.replace('_', ' ')}\n"
            f"Pred: {prediction.replace('_', ' ')}",
            fontsize=8,
            color="darkgreen" if correct else "darkred",
        )
        axis.axis("off")
    for axis in axes.flat[len(selected) :]:
        axis.axis("off")
    figure.suptitle(
        "Selected test predictions - Ground Truth (GT) and Predicted Label",
        fontsize=15,
    )
    _save_figure(
        figure, OUTPUT_DIR / "resnet50_full_finetune_examples.png"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="Full Food-101 root containing meta/classes.txt and images/.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Existing best checkpoint for resnet50_full_finetune.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        parser.error("--batch-size must be positive and --num-workers non-negative")
    if args.checkpoint and not args.dataset_root:
        parser.error("--checkpoint requires --dataset-root")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries, histories = _load_experiments()
    _plot_training_curves(histories)
    _plot_comparisons(summaries)

    if args.dataset_root:
        if not args.dataset_root.is_dir():
            raise FileNotFoundError(f"Food-101 dataset root not found: {args.dataset_root}")
        _plot_sample_grid(args.dataset_root)
        if args.checkpoint:
            _run_selected_model_inference(
                args.dataset_root,
                args.checkpoint,
                args.batch_size,
                args.num_workers,
            )
        else:
            print("Skipping final-model figures: --checkpoint was not provided")
    else:
        print("Skipping image figures: --dataset-root was not provided")
    if not args.checkpoint:
        print(
            "Skipping confusion matrix and prediction examples: the selected "
            "checkpoint was not provided"
        )


if __name__ == "__main__":
    main()
