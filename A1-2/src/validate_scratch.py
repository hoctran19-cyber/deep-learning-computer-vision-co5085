"""Kiểm tra khởi tạo từ đầu cho mô hình Food-101, không tải trọng số.

Chương trình kiểm tra classifier, tham số và đầu ra của ResNet-50 và ViT-B/16
trên dữ liệu ngẫu nhiên; không huấn luyện hoặc tải bộ dữ liệu Food-101.
"""

import sys

import torch
from torch import nn
from torchvision.models.resnet import ResNet
from torchvision.models.vision_transformer import VisionTransformer

from models import build_model, count_parameters


def validate_model(model_name: str, device: torch.device) -> bool:
    """Kiểm tra một mô hình khởi tạo ngẫu nhiên và in kết quả.

    Trả về ``True`` nếu head có 101 lớp, mọi tham số đều hữu hạn và có thể
    huấn luyện, và forward pass trả về tensor hữu hạn có kích thước (2, 101).
    """
    total: int | None = None
    trainable: int | None = None
    output_shape: tuple[int, ...] | None = None
    error_message: str | None = None
    try:
        model = build_model(model_name, num_classes=101, pretrained=False)
        total, trainable = count_parameters(model)

        # Hai kiến trúc lưu head phân loại ở các vị trí khác nhau.
        if model_name == "resnet50" and isinstance(model, ResNet):
            head = model.fc
        elif model_name == "vit_b16" and isinstance(model, VisionTransformer):
            head = model.heads.head
        else:
            raise ValueError(f"Unexpected model type for {model_name!r}")
        if not isinstance(head, nn.Linear) or head.out_features != 101:
            raise ValueError("Classifier head must be linear with 101 outputs")

        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                raise ValueError(f"Parameter {name!r} is frozen")
            if not torch.isfinite(parameter).all().item():
                raise ValueError(f"Parameter {name!r} contains NaN or Inf")

        # eval và inference_mode chỉ phục vụ kiểm tra; không huấn luyện.
        model.to(device)
        model.eval()
        with torch.inference_mode():
            output = model(torch.randn(2, 3, 224, 224, device=device))
        output_shape = tuple(output.shape)
        if output_shape != (2, 101):
            raise ValueError(
                f"Expected output shape (2, 101), got {output_shape}"
            )
        if not torch.isfinite(output).all().item():
            raise ValueError("Output contains NaN or Inf")
    except Exception as error:
        # Ghi nhận lỗi riêng để vẫn tiếp tục kiểm tra mô hình còn lại.
        error_message = f"{type(error).__name__}: {error}"

    print(f"\nModel: {model_name}")
    print("Initialization mode: from scratch (pretrained=False)")
    print(
        f"Total parameters: {total:,}"
        if total is not None else "Total parameters: unavailable"
    )
    print(
        f"Trainable parameters: {trainable:,}"
        if trainable is not None else "Trainable parameters: unavailable"
    )
    print(
        f"Output shape: {output_shape}"
        if output_shape is not None else "Output shape: unavailable"
    )
    print(f"Validation status: {'FAILED' if error_message else 'PASSED'}")
    if error_message:
        print(f"Error ({model_name}): {error_message}", file=sys.stderr)
    return error_message is None


def main() -> int:
    """Kiểm tra cả hai mô hình; trả mã khác 0 nếu có mô hình thất bại."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    # Kiểm tra cả hai mô hình trước khi tổng kết, kể cả khi một mô hình lỗi.
    results = [
        validate_model(model_name, device)
        for model_name in ("resnet50", "vit_b16")
    ]
    passed = all(results)
    print("\nVALIDATION PASSED" if passed else "\nVALIDATION FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
