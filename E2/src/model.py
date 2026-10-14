"""Mô hình phân loại ảnh nhỏ dùng tokenization và MSA."""

from __future__ import annotations

import torch
from torch import nn

from .attention import build_attention
from .tokenizers import build_tokenizer


class TransformerEncoderBlock(nn.Module):
    """Encoder block tối giản: LayerNorm -> MSA -> residual -> FFN -> residual."""

    def __init__(self, embedding_dim: int, num_heads: int, attention_type: str,
                 mlp_ratio: float = 2.0, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_dim = int(embedding_dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.attention = build_attention(attention_type, embedding_dim, num_heads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens + self.dropout(self.attention(self.norm1(tokens)))
        tokens = tokens + self.dropout(self.ffn(self.norm2(tokens)))
        return tokens


class TransformerImageClassifier(nn.Module):
    """Image classifier theo luồng Image -> Tokenization -> MSA -> CLS head."""

    def __init__(self, input_shape: tuple[int, int, int], num_classes: int,
                 tokenizer: str = "patch", attention_type: str = "custom",
                 embedding_dim: int = 64, num_heads: int = 4, num_blocks: int = 2,
                 patch_size: int = 7, stem_channels: int = 32, dropout: float = 0.1,
                 mlp_ratio: float = 2.0) -> None:
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        self.tokenizer = build_tokenizer(tokenizer, input_shape, embedding_dim, patch_size, stem_channels)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, self.tokenizer.num_tokens + 1, embedding_dim))
        self.pos_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embedding_dim, num_heads, attention_type, mlp_ratio, dropout)
            for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, num_classes)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(images)
        batch_size = tokens.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)
        tokens = self.pos_dropout(tokens + self.position_embedding[:, :tokens.shape[1]])
        for block in self.blocks:
            tokens = block(tokens)
        cls_output = self.norm(tokens[:, 0])
        return self.head(cls_output)


def build_model(config: dict, input_shape: tuple[int, int, int], num_classes: int) -> TransformerImageClassifier:
    """Tạo mô hình từ cấu hình thí nghiệm."""
    model_config = config.get("model", {})
    return TransformerImageClassifier(
        input_shape=input_shape,
        num_classes=num_classes,
        tokenizer=model_config.get("tokenizer", "patch"),
        attention_type=model_config.get("attention_type", "custom"),
        embedding_dim=int(model_config.get("embedding_dim", 64)),
        num_heads=int(model_config.get("num_heads", 4)),
        num_blocks=int(model_config.get("num_blocks", 2)),
        patch_size=int(model_config.get("patch_size", 7)),
        stem_channels=int(model_config.get("stem_channels", 32)),
        dropout=float(model_config.get("dropout", 0.1)),
        mlp_ratio=float(model_config.get("mlp_ratio", 2.0)),
    )
