"""Model builders for the Food-101 experiments."""

from torch import nn
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models.resnet import ResNet


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
