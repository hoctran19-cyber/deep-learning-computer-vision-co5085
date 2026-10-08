"""Small PyTorch classifiers required by Exercise E1."""

from typing import Any, Sequence

import torch
from torch import Tensor, nn


class SoftmaxClassifier(nn.Module):
    """Flattened input followed by one linear classification layer."""

    def __init__(self, input_shape: Sequence[int], num_classes: int) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(_input_size(input_shape), num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.flatten(x))


class MLPClassifier(nn.Module):
    """Flattened input followed by configurable fully-connected layers."""

    def __init__(self, input_shape: Sequence[int], num_classes: int,
                 hidden_dims: Sequence[int] = (128, 64), dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Flatten()]
        in_features = _input_size(input_shape)
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(in_features, hidden_dim), nn.ReLU()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)


class CNNClassifier(nn.Module):
    """Two convolution/pooling blocks and a dynamically sized head."""

    def __init__(self, input_shape: Sequence[int], num_classes: int,
                 channels: Sequence[int] = (32, 64), dropout: float = 0.0) -> None:
        super().__init__()
        first, second = channels
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], first, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(first, second, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        with torch.no_grad():
            feature_size = self.features(torch.zeros(1, *input_shape)).flatten(1).shape[1]
        head: list[nn.Module] = [nn.Flatten(), nn.Linear(feature_size, num_classes)]
        if dropout > 0:
            head.insert(1, nn.Dropout(dropout))
        self.classifier = nn.Sequential(*head)

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.features(x))


def _input_size(input_shape: Sequence[int]) -> int:
    size = 1
    for dimension in input_shape:
        size *= dimension
    return size


def get_model(name: str, config: dict[str, Any], input_shape: Sequence[int] = (1, 28, 28),
              num_classes: int = 10) -> nn.Module:
    """Build a model from configuration using ``softmax``, ``mlp``, or ``cnn``."""
    model_config = config.get("model", {})
    model_name = name.lower()
    kwargs = {"dropout": float(model_config.get("dropout", 0.0))}
    if model_name == "softmax":
        return SoftmaxClassifier(input_shape, num_classes)
    if model_name == "mlp":
        return MLPClassifier(input_shape, num_classes, model_config.get("mlp_hidden_dims", [128, 64]), **kwargs)
    if model_name == "cnn":
        return CNNClassifier(input_shape, num_classes, model_config.get("cnn_channels", [32, 64]), **kwargs)
    raise ValueError("Unknown model. Choose softmax, mlp, or cnn.")
