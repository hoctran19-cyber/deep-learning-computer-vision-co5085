import torch

from src.model import TransformerImageClassifier
from src.tokenizers import CNNStemTokenizer, PatchTokenizer, RowTokenizer


def test_tokenizer_shapes() -> None:
    images = torch.randn(4, 1, 28, 28)
    patch = PatchTokenizer((1, 28, 28), patch_size=7, embedding_dim=32)
    row = RowTokenizer((1, 28, 28), embedding_dim=32)
    cnn = CNNStemTokenizer((1, 28, 28), embedding_dim=32)
    assert patch(images).shape == (4, 16, 32)
    assert row(images).shape == (4, 28, 32)
    assert cnn(images).shape == (4, 49, 32)


def test_classifier_forward_backward_custom_attention() -> None:
    model = TransformerImageClassifier(
        input_shape=(1, 28, 28),
        num_classes=10,
        tokenizer="patch",
        attention_type="custom",
        embedding_dim=32,
        num_heads=4,
        num_blocks=1,
    )
    images = torch.randn(2, 1, 28, 28)
    targets = torch.tensor([0, 1])
    logits = model(images)
    assert logits.shape == (2, 10)
    loss = torch.nn.CrossEntropyLoss()(logits, targets)
    loss.backward()
    assert model.head.weight.grad is not None


def test_classifier_forward_torch_attention_row_tokenizer() -> None:
    model = TransformerImageClassifier(
        input_shape=(1, 28, 28),
        num_classes=10,
        tokenizer="row",
        attention_type="torch",
        embedding_dim=32,
        num_heads=4,
        num_blocks=1,
    )
    logits = model(torch.randn(2, 1, 28, 28))
    assert logits.shape == (2, 10)
