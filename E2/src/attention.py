"""Các module Multi-Head Self-Attention dùng trong E2."""

from __future__ import annotations

import math

import torch
from torch import nn


class CustomMultiHeadSelfAttention(nn.Module):
    """Tự triển khai Multi-Head Self-Attention từ các phép toán cơ bản.

    Đầu vào và đầu ra đều có shape ``(batch_size, num_tokens, embedding_dim)``.
    Công thức attention được dùng là::

        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V

    Module này không dùng ``nn.MultiheadAttention`` hay lớp Transformer dựng sẵn.
    """

    def __init__(self, embedding_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        # Ba phép chiếu tuyến tính tạo Query, Key và Value từ cùng chuỗi token.
        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        # Chiếu đầu ra sau khi ghép các head để trộn thông tin giữa các head.
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Đổi ``(B, T, E)`` thành ``(B, H, T, D)`` để tính attention từng head."""
        batch_size, num_tokens, embedding_dim = x.shape
        if embedding_dim != self.embedding_dim:
            raise ValueError(f"Expected embedding_dim={self.embedding_dim}, got {embedding_dim}")
        return x.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Tính self-attention cho chuỗi token đầu vào."""
        if x.ndim != 3:
            raise ValueError("Input must have shape (batch_size, num_tokens, embedding_dim)")
        if x.shape[-1] != self.embedding_dim:
            raise ValueError(f"Expected last dimension {self.embedding_dim}, got {x.shape[-1]}")

        batch_size, num_tokens, _ = x.shape

        # 1. Tạo Query, Key, Value rồi tách thành nhiều attention heads.
        query = self._split_heads(self.q_proj(x))
        key = self._split_heads(self.k_proj(x))
        value = self._split_heads(self.v_proj(x))

        # 2. Tính QK^T và chia cho sqrt(d_k) để tránh softmax quá sắc.
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # 3. Softmax trên chiều token nguồn để tạo trọng số attention.
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        # 4. Tổng có trọng số các vector Value trong từng head.
        attended = torch.matmul(attention_weights, value)

        # 5. Ghép các head về lại embedding_dim ban đầu rồi chiếu đầu ra.
        attended = attended.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.embedding_dim)
        return self.out_proj(attended)


class TorchMultiHeadSelfAttention(nn.Module):
    """Wrapper dùng ``nn.MultiheadAttention`` với cùng giao diện batch-first."""

    def __init__(self, embedding_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Tính self-attention bằng implementation có sẵn của PyTorch."""
        if x.ndim != 3:
            raise ValueError("Input must have shape (batch_size, num_tokens, embedding_dim)")
        output, _ = self.attention(x, x, x, need_weights=False)
        return output


def build_attention(kind: str, embedding_dim: int, num_heads: int, dropout: float = 0.0) -> nn.Module:
    """Tạo module attention theo tên cấu hình."""
    normalized = kind.lower()
    if normalized == "custom":
        return CustomMultiHeadSelfAttention(embedding_dim, num_heads, dropout)
    if normalized == "torch":
        return TorchMultiHeadSelfAttention(embedding_dim, num_heads, dropout)
    raise ValueError("attention kind must be 'custom' or 'torch'")
