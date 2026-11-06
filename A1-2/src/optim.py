"""Tạo optimizer và scheduler cho các thí nghiệm Food-101."""

import math
from collections.abc import Mapping
from numbers import Real
from typing import Any

from torch import nn
from torch.optim import AdamW, SGD, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler, StepLR
from torchvision.models.resnet import ResNet
from torchvision.models.vision_transformer import VisionTransformer


REGIMES = {
    "from_scratch", "frozen_backbone", "partial_finetuning", "full_finetuning"
}


def _section(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """Lấy một mục cấu hình bắt buộc có dạng mapping."""
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a configuration mapping")
    return value


def _number(config: Mapping[str, Any], key: str) -> float:
    """Lấy giá trị số hữu hạn, không chấp nhận boolean hoặc chuỗi."""
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{key} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{key} must be a finite number")
    return result


def _positive_integer(config: Mapping[str, Any], key: str) -> int:
    """Lấy một số nguyên dương."""
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return value


def build_optimizer(model: nn.Module, config: Mapping[str, Any]) -> Optimizer:
    """Tạo optimizer từ cấu hình đầy đủ và trạng thái gradient hiện tại.

    Đặt ``training.regime`` trùng với chế độ truyền vào
    ``configure_finetuning`` trước khi gọi. Chế độ from-scratch dùng
    ``scratch_learning_rate`` cho cả hai nhóm; các chế độ khác dùng
    ``head_learning_rate`` và ``backbone_learning_rate``. Tham số bị đóng
    băng được loại bỏ. Cần tạo lại optimizer nếu trạng thái huấn luyện đổi.
    """
    training = _section(config, "training")
    name = training.get("optimizer")
    if name not in ("AdamW", "SGD"):
        raise ValueError(f"Unsupported optimizer: {name!r}; use AdamW or SGD")
    regime = training.get("regime")
    if not isinstance(regime, str) or regime not in REGIMES:
        raise ValueError(f"Unsupported training regime: {regime!r}")
    weight_decay = _number(training, "weight_decay")
    if weight_decay < 0:
        raise ValueError("weight_decay must be nonnegative")
    if regime == "from_scratch":
        head_lr = backbone_lr = _number(training, "scratch_learning_rate")
    else:
        head_lr = _number(training, "head_learning_rate")
        backbone_lr = _number(training, "backbone_learning_rate")
    if head_lr <= 0 or backbone_lr <= 0:
        raise ValueError("learning rates must be positive")

    # Head của hai kiến trúc nằm ở các thuộc tính khác nhau.
    if isinstance(model, ResNet):
        head = model.fc
    elif isinstance(model, VisionTransformer):
        head = model.heads
    else:
        raise ValueError("Unsupported model type; expected ResNet or VisionTransformer")
    # Dùng danh tính tham số để phân nhóm mà không phụ thuộc tên tham số.
    head_ids = {id(parameter) for parameter in head.parameters()}
    head_params = []
    backbone_params = []
    for parameter in model.parameters():
        if parameter.requires_grad:
            if id(parameter) in head_ids:
                head_params.append(parameter)
            else:
                backbone_params.append(parameter)
    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": backbone_lr})
    if head_params:
        groups.append({"params": head_params, "lr": head_lr})
    if not groups:
        raise ValueError("Model has no trainable parameters")

    if name == "AdamW":
        return AdamW(groups, weight_decay=weight_decay)
    momentum = _number(training, "momentum")
    if not 0 <= momentum < 1:
        raise ValueError("momentum must be in [0, 1)")
    return SGD(groups, momentum=momentum, weight_decay=weight_decay)


def build_scheduler(
    optimizer: Optimizer, config: Mapping[str, Any]
) -> LRScheduler:
    """Tạo scheduler theo cấu hình; gọi step một lần sau mỗi epoch."""
    scheduler = _section(_section(config, "training"), "scheduler")
    name = scheduler.get("name")
    if name == "CosineAnnealingLR":
        options = _section(scheduler, "cosine_annealing")
        t_max = _positive_integer(options, "T_max")
        eta_min = _number(options, "eta_min")
        if eta_min < 0 or any(
            eta_min > group["lr"] for group in optimizer.param_groups
        ):
            raise ValueError(
                "eta_min must be nonnegative and no greater than any group learning rate"
            )
        return CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
    if name == "StepLR":
        options = _section(scheduler, "step")
        step_size = _positive_integer(options, "step_size")
        gamma = _number(options, "gamma")
        if not 0 < gamma <= 1:
            raise ValueError("gamma must be in (0, 1]")
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    raise ValueError(
        f"Unsupported scheduler: {name!r}; use CosineAnnealingLR or StepLR"
    )
