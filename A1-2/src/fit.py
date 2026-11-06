"""Kết nối train, validation, checkpoint và dừng sớm."""

import json
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, TypedDict

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

if __package__:
    from .checkpoint import save_checkpoint
    from .early_stopping import EarlyStopping
    from .evaluate import evaluate_one_epoch
    from .train import train_one_epoch
else:
    from checkpoint import save_checkpoint
    from early_stopping import EarlyStopping
    from evaluate import evaluate_one_epoch
    from train import train_one_epoch


class FitResult(TypedDict):
    """Thông tin checkpoint được chọn và kết quả dừng huấn luyện."""

    best_epoch: int
    best_macro_f1: float
    best_accuracy: float
    best_checkpoint: Path
    epochs_completed: int
    stopped_early: bool


def _validate_dataset_split(dataloader: object, expected: str) -> None:
    """Từ chối split sai khi dataset hoặc wrapper công khai thuộc tính split."""
    dataset = getattr(dataloader, "dataset", None)
    visited: set[int] = set()

    def check(item: object) -> None:
        if id(item) in visited:
            return
        visited.add(id(item))
        split = getattr(item, "split", None)
        if split is not None and split != expected:
            raise ValueError(f"{expected} dataloader has invalid dataset split {split!r}")
        nested = getattr(item, "dataset", None)
        if nested is not None:
            check(nested)
        children = getattr(item, "datasets", None)
        if children is not None:
            for child in children:
                check(child)

    if dataset is not None:
        check(dataset)


def fit(
    model: nn.Module,
    train_dataloader: Iterable[tuple[Tensor, Tensor]],
    validation_dataloader: Iterable[tuple[Tensor, Tensor]],
    criterion: Callable[[Tensor, Tensor], Tensor],
    optimizer: Optimizer,
    device: torch.device | str,
    config: Mapping[str, Any],
    checkpoint_path: Path,
    log_path: Path,
    scheduler: LRScheduler | None = None,
) -> FitResult:
    """Huấn luyện và chọn checkpoint chỉ từ chỉ số validation.

    Caller chuẩn bị model trên device, optimizer và hai DataLoader có thể
    duyệt lại mỗi epoch. Caller phải bảo đảm validation không chứa test;
    split được kiểm tra khi dataset công khai thuộc tính ``split``.
    Log JSONL được nối vào log_path, checkpoint tốt nhất được thay thế nguyên
    tử. Scheduler tùy chọn được bước sau mỗi epoch thành công.
    """
    training = config.get("training")
    if not isinstance(training, Mapping):
        raise ValueError("training must be a configuration mapping")
    max_epochs = training.get("max_epochs")
    if isinstance(max_epochs, bool) or not isinstance(max_epochs, int) or max_epochs <= 0:
        raise ValueError("max_epochs must be a positive integer")
    if training.get("selection_metric") != "macro_f1" or training.get("tie_break_metric") != "accuracy":
        raise ValueError("selection must use macro_f1 and accuracy")
    stopping = EarlyStopping(
        patience=training.get("early_stopping_patience"),
        min_delta=training.get("early_stopping_min_delta"),
    )
    checkpoint_path = Path(checkpoint_path)
    log_path = Path(log_path)
    if checkpoint_path.resolve() == log_path.resolve():
        raise ValueError("checkpoint and log paths must differ")
    if scheduler is not None and scheduler.optimizer is not optimizer:
        raise ValueError("scheduler must use the supplied optimizer")
    _validate_dataset_split(train_dataloader, "train")
    _validate_dataset_split(validation_dataloader, "validation")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    completed = 0
    for epoch in range(1, max_epochs + 1):
        training_metrics = train_one_epoch(
            model, train_dataloader, criterion, optimizer, device
        )
        validation_metrics = evaluate_one_epoch(
            model, validation_dataloader, criterion, device
        )
        improved = stopping.update(
            epoch, validation_metrics["macro_f1"], validation_metrics["accuracy"]
        )
        if scheduler is not None:
            scheduler.step()
        if improved:
            save_checkpoint(checkpoint_path, {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "validation_metrics": dict(validation_metrics),
            })
        record = {
            "epoch": epoch,
            "train": training_metrics,
            "validation": validation_metrics,
            "best_epoch": stopping.best_epoch,
            "checkpoint_saved": improved,
            "bad_epochs": stopping.bad_epochs,
            "stopped_early": stopping.should_stop,
        }
        with log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, allow_nan=False) + "\n")
        completed = epoch
        if stopping.should_stop:
            break
    if stopping.best_epoch is None or stopping.best_score is None or stopping.best_accuracy is None:
        raise RuntimeError("no validation checkpoint was selected")
    return {
        "best_epoch": stopping.best_epoch,
        "best_macro_f1": stopping.best_score,
        "best_accuracy": stopping.best_accuracy,
        "best_checkpoint": checkpoint_path,
        "epochs_completed": completed,
        "stopped_early": stopping.should_stop,
    }
