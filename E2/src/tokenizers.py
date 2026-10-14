"""Các phương pháp tokenization ảnh cho Transformer nhỏ."""

from __future__ import annotations

import torch
from torch import nn


class PatchTokenizer(nn.Module):
    """Chia ảnh thành các patch không chồng lấp rồi chiếu từng patch thành token."""

    def __init__(self, input_shape: tuple[int, int, int], patch_size: int, embedding_dim: int) -> None:
        super().__init__()
        channels, height, width = input_shape
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError("height and width must be divisible by patch_size")
        self.input_shape = input_shape
        self.patch_size = patch_size
        self.num_tokens = (height // patch_size) * (width // patch_size)
        self.projection = nn.Linear(channels * patch_size * patch_size, embedding_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Trả về token shape ``(B, num_patches, embedding_dim)``."""
        batch_size, channels, height, width = images.shape
        patch = self.patch_size
        patches = images.unfold(2, patch, patch).unfold(3, patch, patch)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(batch_size, -1, channels * patch * patch)
        return self.projection(patches)


class RowTokenizer(nn.Module):
    """Mỗi hàng pixel của ảnh là một token riêng biệt.

    Với Fashion-MNIST, ảnh ``1x28x28`` tạo 28 token, mỗi token là vector gồm
    toàn bộ pixel trên một hàng. Cách này khác patch tokenization vì token giữ
    cấu trúc theo hàng ngang thay vì ô vuông cục bộ.
    """

    def __init__(self, input_shape: tuple[int, int, int], embedding_dim: int) -> None:
        super().__init__()
        channels, height, width = input_shape
        self.input_shape = input_shape
        self.num_tokens = height
        self.projection = nn.Linear(channels * width, embedding_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Trả về token shape ``(B, height, embedding_dim)``."""
        batch_size, channels, height, width = images.shape
        rows = images.permute(0, 2, 1, 3).contiguous().view(batch_size, height, channels * width)
        return self.projection(rows)


class CNNStemTokenizer(nn.Module):
    """Dùng CNN nhỏ tạo feature map rồi biến từng vị trí không gian thành token."""

    def __init__(self, input_shape: tuple[int, int, int], embedding_dim: int, stem_channels: int = 32) -> None:
        super().__init__()
        channels, height, width = input_shape
        self.stem = nn.Sequential(
            nn.Conv2d(channels, stem_channels, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(stem_channels, embedding_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        out_height = (height + 1) // 2
        out_width = (width + 1) // 2
        out_height = (out_height + 1) // 2
        out_width = (out_width + 1) // 2
        self.num_tokens = out_height * out_width
        self.input_shape = input_shape

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Trả về token shape ``(B, H_feature * W_feature, embedding_dim)``."""
        features = self.stem(images)
        return features.flatten(2).transpose(1, 2)


def build_tokenizer(name: str, input_shape: tuple[int, int, int], embedding_dim: int,
                    patch_size: int = 7, stem_channels: int = 32) -> nn.Module:
    """Tạo tokenizer ảnh theo tên cấu hình."""
    normalized = name.lower()
    if normalized == "patch":
        return PatchTokenizer(input_shape, patch_size, embedding_dim)
    if normalized == "row":
        return RowTokenizer(input_shape, embedding_dim)
    if normalized == "cnn_stem":
        return CNNStemTokenizer(input_shape, embedding_dim, stem_channels)
    raise ValueError("tokenizer must be 'patch', 'row', or 'cnn_stem'")
