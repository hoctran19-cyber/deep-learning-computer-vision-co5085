"""Validate actual ImageNet pretrained model loading and CPU/CUDA inference.

The first run may download Torchvision weights; no Food-101 images are needed.
The Food-101 classifier head is newly initialized, not pretrained.
"""

import sys

import torch
from torch import nn
from torchvision.models.resnet import ResNet
from torchvision.models.vision_transformer import VisionTransformer

from models import build_model, count_parameters


def validate_model(model_name: str, device: torch.device) -> bool:
    """Build and validate one model, reporting any failure independently."""
    pretrained_status = "not loaded"
    total: int | None = None
    trainable: int | None = None
    output_shape: tuple[int, ...] | None = None
    error_message: str | None = None
    try:
        model = build_model(model_name, num_classes=101, pretrained=True)
        pretrained_status = "loaded (ImageNet backbone; new classifier head)"
        total, trainable = count_parameters(model)

        if model_name == "resnet50" and isinstance(model, ResNet):
            head = model.fc
        elif model_name == "vit_b16" and isinstance(model, VisionTransformer):
            head = model.heads.head
        else:
            raise ValueError(f"Unexpected model type for {model_name!r}")
        if not isinstance(head, nn.Linear) or head.out_features != 101:
            raise ValueError("Classifier head must be linear with 101 outputs")

        model.to(device)
        model.eval()
        with torch.inference_mode():
            output = model(torch.randn(2, 3, 224, 224, device=device))
        output_shape = tuple(output.shape)
        if output_shape != (2, 101):
            raise ValueError(f"Expected output shape (2, 101), got {output_shape}")
        if not torch.isfinite(output).all().item():
            raise ValueError("Output contains non-finite values")
    except Exception as error:
        error_message = f"{type(error).__name__}: {error}"

    print(f"\nModel: {model_name}")
    print(f"Pretrained status: {pretrained_status}")
    print(f"Total parameters: {total:,}" if total is not None else "Total parameters: unavailable")
    print(
        f"Trainable parameters: {trainable:,}"
        if trainable is not None else "Trainable parameters: unavailable"
    )
    print(f"Output shape: {output_shape if output_shape is not None else 'unavailable'}")
    print(f"Validation status: {'FAILED' if error_message else 'PASSED'}")
    if error_message:
        print(f"Error: {error_message}", file=sys.stderr)
    return error_message is None


def main() -> int:
    """Validate both pretrained models and return a nonzero status on failure."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    results = [
        validate_model(model_name, device)
        for model_name in ("resnet50", "vit_b16")
    ]
    passed = all(results)
    print("\nVALIDATION PASSED" if passed else "\nVALIDATION FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
