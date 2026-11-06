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
