"""Đánh giá một epoch trên tập validation để lựa chọn mô hình."""

from collections.abc import Callable, Iterable
from typing import TypedDict

import torch
from torch import Tensor, nn


class ValidationMetrics(TypedDict):
    """Loss trung bình theo mẫu, accuracy, macro-F1 và số mẫu."""

    loss: float
    accuracy: float
    macro_f1: float
    num_samples: int


def evaluate_one_epoch(
    model: nn.Module,
    dataloader: Iterable[tuple[Tensor, Tensor]],
    criterion: Callable[[Tensor, Tensor], Tensor],
    device: torch.device | str,
) -> ValidationMetrics:
    """Đánh giá validation không tính gradient và khôi phục chế độ mô hình.

    Model phải được chuyển sang device trước khi gọi. Criterion phải trả về
    loss vô hướng trung bình theo mọi mẫu trong batch. Accuracy và macro-F1
    nằm trong [0, 1]; macro-F1 tính cả lớp không có mẫu hoặc dự đoán.
    """
    if not callable(criterion):
        raise ValueError("criterion must be callable")
    if getattr(criterion, "reduction", "mean") != "mean":
        raise ValueError("criterion must use mean reduction")
    try:
        batches = iter(dataloader)
    except TypeError as error:
        raise ValueError("dataloader must be an iterable of batches") from error

    # Giữ cả chế độ riêng của các mô-đun con, ví dụ BatchNorm bị cố định.
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        loss_sum = 0.0
        num_samples = 0
        num_classes: int | None = None
        confusion: Tensor | None = None
        with torch.inference_mode():
            for batch_index, batch in enumerate(batches, start=1):
                if not isinstance(batch, (tuple, list)) or len(batch) != 2:
                    raise ValueError(
                        f"batch {batch_index}: expected (images, labels)"
                    )
                images, labels = batch
                if not isinstance(images, Tensor) or not isinstance(labels, Tensor):
                    raise ValueError(
                        f"batch {batch_index}: images and labels must be tensors"
                    )
                if images.ndim < 1 or labels.ndim != 1:
                    raise ValueError(
                        f"batch {batch_index}: invalid image or label dimensions"
                    )
                batch_size = labels.shape[0]
                if batch_size == 0 or images.shape[0] != batch_size:
                    raise ValueError(
                        f"batch {batch_index}: empty or mismatched batch"
                    )
                if labels.dtype != torch.long:
                    raise ValueError(
                        f"batch {batch_index}: labels must have dtype torch.int64"
                    )
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(images)
                if (
                    not isinstance(logits, Tensor)
                    or logits.ndim != 2
                    or logits.shape[0] != batch_size
                    or logits.shape[1] == 0
                ):
                    raise ValueError(
                        f"batch {batch_index}: expected logits shape [batch, classes]"
                    )
                if num_classes is None:
                    num_classes = logits.shape[1]
                    confusion = torch.zeros(
                        (num_classes, num_classes), dtype=torch.long, device=device
                    )
                elif logits.shape[1] != num_classes:
                    raise ValueError(
                        f"batch {batch_index}: number of classes changed"
                    )
                if not torch.isfinite(logits).all().item():
                    raise ValueError(f"batch {batch_index}: logits must be finite")
                if ((labels < 0) | (labels >= num_classes)).any().item():
                    raise ValueError(f"batch {batch_index}: label outside class range")
                loss = criterion(logits, labels)
                if not isinstance(loss, Tensor) or loss.ndim != 0:
                    raise ValueError(
                        f"batch {batch_index}: criterion must return scalar loss"
                    )
                if not torch.isfinite(loss).item():
                    raise ValueError(f"batch {batch_index}: loss must be finite")
                predictions = logits.argmax(dim=1)
                if confusion is None:
                    raise RuntimeError("confusion matrix was not initialized")
                confusion += torch.bincount(
                    labels * num_classes + predictions,
                    minlength=num_classes * num_classes,
                ).reshape(num_classes, num_classes)
                loss_sum += loss.item() * batch_size
                num_samples += batch_size

        if num_samples == 0 or confusion is None:
            raise ValueError("dataloader yielded no samples")
        true_positives = confusion.diag()
        denominator = confusion.sum(dim=0) + confusion.sum(dim=1)
        f1 = torch.zeros_like(denominator, dtype=torch.float64)
        valid = denominator > 0
        f1[valid] = (
            2 * true_positives[valid].to(torch.float64) / denominator[valid]
        )
        return {
            "loss": loss_sum / num_samples,
            "accuracy": true_positives.sum().item() / num_samples,
            "macro_f1": f1.mean().item(),
            "num_samples": num_samples,
        }
    finally:
        for module, training in modes:
            module.training = training
