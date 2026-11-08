"""Huấn luyện mô hình phân loại trong một epoch."""

from collections.abc import Callable, Iterable
from typing import TypedDict

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torchvision.models.resnet import ResNet


class EpochMetrics(TypedDict):
    """Loss trung bình theo mẫu, độ chính xác top-1 và số mẫu đã xử lý."""

    loss: float
    accuracy: float
    num_samples: int


def _new_grad_scaler():
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda")
    return torch.cuda.amp.GradScaler()


def _set_training_mode(model: nn.Module) -> None:
    """Keep frozen ResNet backbone BatchNorm buffers fixed during training."""
    model.train()
    if isinstance(model, ResNet) and not any(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if not name.startswith("fc.")
    ):
        for name, module in model.named_modules():
            if not name.startswith("fc.") and isinstance(
                module, nn.modules.batchnorm._BatchNorm
            ):
                module.eval()


def train_one_epoch(
    model: nn.Module,
    dataloader: Iterable[tuple[Tensor, Tensor]],
    criterion: Callable[[Tensor, Tensor], Tensor],
    optimizer: Optimizer,
    device: torch.device | str,
    use_amp: bool = False,
    scaler: "torch.amp.GradScaler | None" = None,
) -> EpochMetrics:
    """Huấn luyện một epoch; accuracy là tỷ lệ trong khoảng [0, 1].

    Caller phải chuyển model sang device trước khi gọi. Mỗi batch gồm ảnh
    tensor và nhãn int64 một chiều. Criterion phải trả về loss vô hướng trung
    bình theo mẫu, không dùng reduction='sum' hoặc 'none'. Với loss có trọng
    số hoặc ignore_index, caller cần đảm bảo kết quả vẫn là trung bình theo
    tất cả mẫu trong batch để thống kê loss theo epoch có ý nghĩa.

    ``use_amp`` bật mixed precision chỉ khi device là CUDA và CUDA khả dụng;
    ngược lại AMP tự tắt. ``scaler`` tùy chọn cho phép giữ trạng thái scale
    giữa các epoch; nếu thiếu, một GradScaler mới được tạo cho epoch này.
    """
    if not isinstance(use_amp, bool):
        raise ValueError("use_amp must be a boolean")
    device_type = torch.device(device).type
    amp_enabled = use_amp and device_type == "cuda" and torch.cuda.is_available()
    if amp_enabled and scaler is None:
        scaler = _new_grad_scaler()
    if not amp_enabled:
        scaler = None
    if not callable(criterion):
        raise ValueError("criterion must be callable")
    reduction = getattr(criterion, "reduction", None)
    if reduction != "mean":
        raise ValueError("criterion must use mean reduction (declare reduction='mean')")
    try:
        batches = iter(dataloader)
    except TypeError as error:
        raise ValueError("dataloader must be an iterable of batches") from error

    _set_training_mode(model)
    loss_sum = 0.0
    correct = 0
    num_samples = 0

    for batch_index, batch in enumerate(batches, start=1):
        if not isinstance(batch, (tuple, list)) or len(batch) != 2:
            raise ValueError(f"batch {batch_index}: expected (images, labels)")
        images, labels = batch
        if not isinstance(images, Tensor) or not isinstance(labels, Tensor):
            raise ValueError(f"batch {batch_index}: images and labels must be tensors")
        if images.ndim < 1 or labels.ndim != 1:
            raise ValueError(f"batch {batch_index}: invalid image or label dimensions")
        batch_size = labels.shape[0]
        if batch_size == 0 or images.shape[0] != batch_size:
            raise ValueError(f"batch {batch_index}: empty or mismatched batch")
        if labels.dtype != torch.long:
            raise ValueError(f"batch {batch_index}: labels must have dtype torch.int64")

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device_type, enabled=amp_enabled):
            logits = model(images)
        if not isinstance(logits, Tensor):
            raise ValueError(f"batch {batch_index}: model output must be a Tensor")
        if logits.ndim != 2:
            raise ValueError(
                f"batch {batch_index}: logits must have shape [batch_size, num_classes], "
                f"got {tuple(logits.shape)}"
            )
        if logits.shape[0] != batch_size:
            raise ValueError(
                f"batch {batch_index}: logits batch size {logits.shape[0]} "
                f"does not match target batch size {batch_size}"
            )
        if logits.shape[1] == 0:
            raise ValueError(f"batch {batch_index}: logits must have at least one class")
        if not torch.isfinite(logits).all().item():
            raise ValueError(f"batch {batch_index}: logits must be finite")
        if ((labels < 0) | (labels >= logits.shape[1])).any().item():
            raise ValueError(
                f"batch {batch_index}: label outside class range "
                f"[0, {logits.shape[1] - 1}]"
            )
        with torch.amp.autocast(device_type=device_type, enabled=amp_enabled):
            loss = criterion(logits, labels)
        if not isinstance(loss, Tensor):
            raise ValueError(f"batch {batch_index}: criterion must return a Tensor")
        if loss.ndim != 0:
            raise ValueError(f"batch {batch_index}: criterion must return scalar loss")
        if not torch.isfinite(loss).item():
            raise ValueError(f"batch {batch_index}: loss must be finite")
        if not loss.requires_grad:
            raise ValueError(
                f"batch {batch_index}: loss must be connected to a gradient graph"
            )
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        loss_sum += loss.detach().item() * batch_size
        correct += (logits.detach().argmax(dim=1) == labels).sum().item()
        num_samples += batch_size

    if num_samples == 0:
        raise ValueError("dataloader yielded no samples")
    return {
        "loss": loss_sum / num_samples,
        "accuracy": correct / num_samples,
        "num_samples": num_samples,
    }
