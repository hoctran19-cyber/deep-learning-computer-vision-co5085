"""Model builders for the Food-101 experiments."""

from torch import nn
from torchvision.models import (
    ResNet50_Weights,
    ViT_B_16_Weights,
    resnet50,
    vit_b_16,
)
from torchvision.models.resnet import ResNet
from torchvision.models.vision_transformer import VisionTransformer


def build_resnet50(num_classes: int = 101, pretrained: bool = False) -> ResNet:
    """Build ResNet-50 with a classification head for ``num_classes``.

    Set ``pretrained`` to load the default ImageNet weights, which may require
    a download. Otherwise, initialize the entire model randomly.
    """
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")

    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_vit_b16(
    num_classes: int = 101, pretrained: bool = False
) -> VisionTransformer:
    """Build ViT-B/16 with a classification head for ``num_classes``.

    Set ``pretrained`` to load the default ImageNet weights, which may require
    a download. Otherwise, initialize the entire model randomly.
    """
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")

    weights = ViT_B_16_Weights.DEFAULT if pretrained else None
    model = vit_b_16(weights=weights)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model


def build_model(
    model_name: str, num_classes: int = 101, pretrained: bool = False
) -> ResNet | VisionTransformer:
    """Build a supported model with the requested head and initialization.

    Supported names are ``resnet50`` and ``vit_b16``. Pretrained weights
    may need to be downloaded on first use.
    """
    if model_name == "resnet50":
        return build_resnet50(num_classes=num_classes, pretrained=pretrained)
    if model_name == "vit_b16":
        return build_vit_b16(num_classes=num_classes, pretrained=pretrained)
    raise ValueError(
        f"Unsupported model name {model_name!r}; "
        "expected 'resnet50' or 'vit_b16'"
    )


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return the number of scalar parameters as (total, trainable)."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total, trainable


def freeze_parameters(module: nn.Module) -> None:
    """Disable gradients for all parameters in a module."""
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def unfreeze_parameters(module: nn.Module) -> None:
    """Enable gradients for all parameters in a module."""
    for parameter in module.parameters():
        parameter.requires_grad_(True)


def configure_finetuning(model: nn.Module, regime: str) -> None:
    """Set trainable parameters for a supported model and training regime.

    ``from_scratch`` and ``full_finetuning`` unfreeze every parameter.
    ``frozen_backbone`` unfreezes only the classifier. ``partial_finetuning``
    unfreezes ResNet's layer4 and fc, or ViT's last encoder block and heads.
    Other parameters, including ViT's final encoder norm, remain frozen.

    This function does not initialize weights or change train/eval mode.
    Build with ``pretrained=False`` for an actual from-scratch experiment.
    Frozen BatchNorm buffers can still update in training mode.
    """
    regimes = {
        "from_scratch", "frozen_backbone",
        "partial_finetuning", "full_finetuning",
    }
    if regime not in regimes:
        raise ValueError(f"Unsupported training regime: {regime!r}")
    if isinstance(model, ResNet):
        classifier = model.fc
        last_block = model.layer4
    elif isinstance(model, VisionTransformer):
        classifier = model.heads
        last_block = model.encoder.layers[-1]
    else:
        raise ValueError(
            "Unsupported model type; expected a ResNet or VisionTransformer"
        )

    if regime in {"from_scratch", "full_finetuning"}:
        unfreeze_parameters(model)
    else:
        freeze_parameters(model)
        unfreeze_parameters(classifier)
        if regime == "partial_finetuning":
            unfreeze_parameters(last_block)
