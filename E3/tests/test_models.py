"""Tests for row-wise recurrent image classifiers."""

import sys
from pathlib import Path

import pytest
import torch
import yaml
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from E3.src.datasets import image_to_row_sequence
from E3.src.models import GRUClassifier, LSTMClassifier, get_model


@pytest.fixture
def config() -> dict:
    """Load the configuration used to construct the default models."""
    with (ROOT / "E3" / "configs" / "config.yaml").open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def test_row_sequence_preserves_pixels_and_order() -> None:
    images = torch.arange(2 * 28 * 28).reshape(2, 1, 28, 28)
    sequence = image_to_row_sequence(images)
    assert sequence.shape == (2, 28, 28)
    assert torch.equal(sequence, images[:, 0, :, :])


@pytest.mark.parametrize("shape", [(2, 28, 28), (2, 3, 28, 28), (2, 1, 0, 28)])
def test_row_sequence_rejects_invalid_shapes(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        image_to_row_sequence(torch.zeros(shape))


@pytest.mark.parametrize(
    ("name", "model_type", "recurrent_type", "parameter_count"),
    [
        ("lstm", LSTMClassifier, nn.LSTM, 214282),
        ("gru", GRUClassifier, nn.GRU, 161034),
    ],
)
def test_default_models_forward_backward_and_factory(
    config: dict, name: str, model_type: type, recurrent_type: type, parameter_count: int
) -> None:
    assert {key: config[key] for key in (
        "seed", "input_size", "hidden_size", "num_layers", "dropout", "num_classes"
    )} == {
        "seed": 42,
        "input_size": 28,
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.2,
        "num_classes": 10,
    }
    model = get_model(name, config)
    assert isinstance(model, model_type)
    assert isinstance(model.recurrent, recurrent_type)
    assert sum(parameter.numel() for parameter in model.parameters()) == parameter_count

    images = torch.randn(2, 1, 28, 28, requires_grad=True)
    logits = model(images)
    assert logits.shape == (2, 10)
    assert torch.isfinite(logits).all()
    logits.square().mean().backward()
    assert images.grad is not None
    assert torch.isfinite(images.grad).all()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize("model_type", [LSTMClassifier, GRUClassifier])
def test_configuration_and_final_hidden_representation(model_type: type) -> None:
    model = model_type(input_size=5, hidden_size=7, num_layers=1, dropout=0.4, num_classes=3)
    assert model.recurrent.dropout == 0
    model.eval()
    images = torch.randn(4, 1, 6, 5)
    output, hidden = model.recurrent(image_to_row_sequence(images))
    final_hidden = hidden[0] if isinstance(hidden, tuple) else hidden
    assert torch.allclose(output[:, -1, :], final_hidden[-1])
    assert torch.allclose(model(images), model.classifier(final_hidden[-1]))
    assert model(images).shape == (4, 3)
    with pytest.raises(ValueError, match="width"):
        model(torch.randn(4, 1, 6, 4))


def test_factory_rejects_unknown_name(config: dict) -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        get_model("unknown", config)


@pytest.mark.parametrize("name", ["LSTM", "GRU"])
def test_factory_case_insensitive_and_custom_settings(name: str) -> None:
    model = get_model(name, {
        "input_size": 5, "hidden_size": 7, "num_layers": 2,
        "dropout": 0.3, "num_classes": 3,
    })
    assert model.recurrent.input_size == 5
    assert model.recurrent.hidden_size == 7
    assert model.recurrent.num_layers == 2
    assert model.recurrent.dropout == 0.3
    assert model(torch.randn(2, 1, 6, 5)).shape == (2, 3)
