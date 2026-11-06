"""Kiểm thử tích hợp pipeline trên dữ liệu ảnh giả lập và CPU."""

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

if __package__:
    from .fit import FitResult, fit
    from .optim import build_optimizer, build_scheduler
else:
    from fit import FitResult, fit
    from optim import build_optimizer, build_scheduler


class SmallCNN(nn.Module):
    """CNN nhỏ dùng riêng cho smoke test, không phải mô hình Food-101."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(4, 3)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Trả logits của ba lớp giả lập."""
        return self.classifier(self.features(images))


def _validate_artifacts(result: FitResult, log_path: Path) -> None:
    """Kiểm tra nội dung checkpoint và metric của mọi epoch."""
    checkpoint = torch.load(
        result["best_checkpoint"], map_location="cpu", weights_only=True
    )
    required = {
        "epoch", "model_state_dict", "optimizer_state_dict",
        "scheduler_state_dict", "validation_metrics",
    }
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        raise ValueError("checkpoint is missing required fields")
    if any(not isinstance(checkpoint[key], dict) for key in required - {"epoch"}):
        raise ValueError("checkpoint state and metrics must be dictionaries")
    if checkpoint["epoch"] != result["best_epoch"]:
        raise ValueError("checkpoint epoch does not match selected epoch")
    with log_path.open(encoding="utf-8") as file:
        records = [json.loads(line) for line in file]
    if len(records) != result["epochs_completed"]:
        raise ValueError("log does not contain exactly one record per epoch")
    for record in records:
        for split, expected_count in (("train", 12), ("validation", 7)):
            metrics = record[split]
            if (
                metrics["num_samples"] != expected_count
                or not math.isfinite(metrics["loss"])
                or metrics["loss"] < 0
                or not math.isfinite(metrics["accuracy"])
                or not 0 <= metrics["accuracy"] <= 1
            ):
                raise ValueError(f"invalid {split} metrics")
            if split == "validation" and (
                not math.isfinite(metrics["macro_f1"])
                or not 0 <= metrics["macro_f1"] <= 1
            ):
                raise ValueError("invalid validation macro_f1")
    selected = records[result["best_epoch"] - 1]
    best = max(
        records,
        key=lambda record: (
            record["validation"]["macro_f1"], record["validation"]["accuracy"]
        ),
    )
    if (
        selected["epoch"] != best["epoch"]
        or not selected["checkpoint_saved"]
        or checkpoint["validation_metrics"] != selected["validation"]
        or result["best_macro_f1"] != selected["validation"]["macro_f1"]
        or result["best_accuracy"] != selected["validation"]["accuracy"]
    ):
        raise ValueError("checkpoint does not match the best validation epoch")


def run_smoke_test(config: dict[str, Any], output_dir: Path) -> FitResult:
    """Chạy pipeline CPU và xác minh artifact, không tải dữ liệu hay weights."""
    config = copy.deepcopy(config)
    training = config["training"]
    training.update(
        regime="from_scratch",
        max_epochs=3,
        early_stopping_patience=1,
        early_stopping_min_delta=1.0,
    )
    checkpoint_path = output_dir / "best.pt"
    log_path = output_dir / "epochs.jsonl"
    if checkpoint_path.exists() or log_path.exists():
        raise FileExistsError("smoke test output files already exist")
    seed = config["project"]["seed"]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed)
        train_data = TensorDataset(
            torch.randn(12, 3, 8, 8, generator=generator),
            torch.randint(0, 3, (12,), generator=generator),
        )
        val_generator = torch.Generator().manual_seed(seed + 1)
        validation_data = TensorDataset(
            torch.randn(7, 3, 8, 8, generator=val_generator),
            torch.randint(0, 3, (7,), generator=val_generator),
        )
        train_loader = DataLoader(
            train_data, batch_size=4, shuffle=True,
            generator=torch.Generator().manual_seed(seed + 2), num_workers=0,
        )
        validation_loader = DataLoader(
            validation_data, batch_size=4, shuffle=False, num_workers=0,
        )
        model = SmallCNN().to("cpu")
        optimizer = build_optimizer(model, config, classifier=model.classifier)
        scheduler = build_scheduler(optimizer, config)
        result = fit(
            model, train_loader, validation_loader, nn.CrossEntropyLoss(),
            optimizer, "cpu", config, checkpoint_path, log_path, scheduler,
        )
    _validate_artifacts(result, log_path)
    if not result["stopped_early"] or result["epochs_completed"] != 2:
        raise ValueError("early stopping did not trigger as expected")
    return result


def main() -> int:
    """Chạy smoke test với cấu hình hiện có và thư mục đầu ra do caller chọn."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config_path = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"
    try:
        with config_path.open(encoding="utf-8") as file:
            config = yaml.safe_load(file)
        result = run_smoke_test(config, args.output_dir)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, yaml.YAMLError) as error:
        print(f"SMOKE TEST FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(f"SMOKE TEST PASSED: best epoch {result['best_epoch']}")
    print(f"Checkpoint: {result['best_checkpoint']}")
    print(f"Log: {args.output_dir / 'epochs.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
