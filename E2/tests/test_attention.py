import torch
import pytest

from src.attention import CustomMultiHeadSelfAttention, TorchMultiHeadSelfAttention


def test_custom_msa_shape_and_backward() -> None:
    module = CustomMultiHeadSelfAttention(embedding_dim=32, num_heads=4)
    inputs = torch.randn(2, 8, 32, requires_grad=True)
    outputs = module(inputs)
    assert outputs.shape == inputs.shape
    outputs.mean().backward()
    assert inputs.grad is not None
    assert module.q_proj.weight.grad is not None


def test_torch_msa_shape_and_backward() -> None:
    module = TorchMultiHeadSelfAttention(embedding_dim=32, num_heads=4)
    inputs = torch.randn(2, 8, 32, requires_grad=True)
    outputs = module(inputs)
    assert outputs.shape == inputs.shape
    outputs.sum().backward()
    assert inputs.grad is not None


def test_custom_msa_requires_divisible_embedding() -> None:
    with pytest.raises(ValueError):
        CustomMultiHeadSelfAttention(embedding_dim=30, num_heads=8)
