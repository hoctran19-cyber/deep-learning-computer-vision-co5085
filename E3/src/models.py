"""Row-wise LSTM and GRU image classifiers.

The factory accepts the top-level mapping in ``configs/config.yaml``. Each
model returns unnormalized class logits and can be moved to any PyTorch device.
"""

from typing import Any, Mapping

from torch import Tensor, nn

from .datasets import image_to_row_sequence


class _SequenceClassifier(nn.Module):
    """Shared row-wise recurrent classifier implementation."""

    def __init__(
        self,
        recurrent_type: type[nn.LSTM] | type[nn.GRU],
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        num_classes: int,
    ) -> None:
        super().__init__()
        if not 0 <= dropout <= 1:
            raise ValueError("dropout must be between 0 and 1.")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")
        self.input_size = input_size
        self.recurrent = recurrent_type(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, images: Tensor) -> Tensor:
        """Return class logits for a batch of grayscale images."""
        sequence = image_to_row_sequence(images)
        if sequence.shape[-1] != self.input_size:
            raise ValueError(
                f"Expected image width {self.input_size}, got {sequence.shape[-1]}."
            )
        output, _ = self.recurrent(sequence)
        return self.classifier(output[:, -1, :])


class LSTMClassifier(_SequenceClassifier):
    """Classify images using the last output of a unidirectional LSTM."""

    def __init__(
        self,
        input_size: int = 28,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        num_classes: int = 10,
    ) -> None:
        super().__init__(nn.LSTM, input_size, hidden_size, num_layers, dropout, num_classes)


class GRUClassifier(_SequenceClassifier):
    """Classify images using the last output of a unidirectional GRU."""

    def __init__(
        self,
        input_size: int = 28,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        num_classes: int = 10,
    ) -> None:
        super().__init__(nn.GRU, input_size, hidden_size, num_layers, dropout, num_classes)


def get_model(name: str, config: Mapping[str, Any]) -> LSTMClassifier | GRUClassifier:
    """Build an LSTM or GRU from a flat configuration mapping.

    Recurrent dropout is applied between layers only, as in PyTorch; it is
    disabled for single-layer models.
    """
    models = {"lstm": LSTMClassifier, "gru": GRUClassifier}
    model_name = name.lower()
    if model_name not in models:
        raise ValueError(f"Unknown model '{name}'. Choose lstm or gru.")
    return models[model_name](
        input_size=config.get("input_size", 28),
        hidden_size=config.get("hidden_size", 128),
        num_layers=config.get("num_layers", 2),
        dropout=config.get("dropout", 0.2),
        num_classes=config.get("num_classes", 10),
    )
