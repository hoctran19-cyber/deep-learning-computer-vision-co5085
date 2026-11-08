"""Kiểm thử vòng huấn luyện một epoch bằng dữ liệu giả lập."""

import sys
import unittest
from unittest import mock
from pathlib import Path

import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from train import train_one_epoch


class TrainingEpochTests(unittest.TestCase):
    """Kiểm tra phép tính metric, cập nhật tham số và dữ liệu không hợp lệ."""

    def setUp(self) -> None:
        self.images = torch.tensor([[3.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        self.labels = torch.tensor([0, 0, 1], dtype=torch.long)
        self.loader = DataLoader(
            TensorDataset(self.images, self.labels), batch_size=2, shuffle=False
        )
        self.model = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.model.weight.copy_(torch.eye(2))
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = SGD(self.model.parameters(), lr=0.0)

    def train(self, batches: object) -> dict[str, float | int]:
        """Chạy vòng huấn luyện trên CPU với các batch được cung cấp."""
        return train_one_epoch(
            self.model, batches, self.criterion, self.optimizer, torch.device("cpu")
        )

    def test_metrics_use_all_samples_including_unequal_last_batch(self) -> None:
        self.model.eval()
        expected_loss = self.criterion(self.images, self.labels).item()
        metrics = self.train(self.loader)
        self.assertTrue(self.model.training)
        self.assertEqual(metrics["num_samples"], 3)
        self.assertAlmostEqual(metrics["loss"], expected_loss, places=6)
        self.assertAlmostEqual(metrics["accuracy"], 1 / 3)

    def test_optimizer_updates_parameters(self) -> None:
        optimizer = SGD(self.model.parameters(), lr=0.1)
        before = self.model.weight.detach().clone()
        metrics = train_one_epoch(
            self.model, self.loader, self.criterion, optimizer, "cpu"
        )
        self.assertEqual(metrics["num_samples"], 3)
        self.assertFalse(torch.equal(before, self.model.weight))
        self.assertEqual(self.model.weight.device.type, "cpu")

    def test_empty_and_noniterable_dataloader(self) -> None:
        for batches in ([], None):
            with self.subTest(batches=batches), self.assertRaisesRegex(
                ValueError, "dataloader"
            ):
                self.train(batches)

    def test_malformed_batches(self) -> None:
        cases = [
            "bad",
            (self.images,),
            (self.images, self.labels, self.labels),
            ([1, 2], self.labels),
            (torch.tensor(1.0), self.labels),
            (self.images, self.labels.unsqueeze(1)),
            (self.images, self.labels[:2]),
            (torch.empty((0, 2)), torch.empty(0, dtype=torch.long)),
            (self.images, self.labels.float()),
            (self.images, torch.tensor([0, 1, 2])),
            (self.images, torch.tensor([0, 1, -1])),
        ]
        for batch in cases:
            with self.subTest(batch=repr(batch)), self.assertRaisesRegex(
                ValueError, "batch 1"
            ):
                self.train([batch])

    def test_invalid_criterion_reduction_and_result(self) -> None:
        for reduction in ("sum", "none"):
            with self.subTest(reduction=reduction), self.assertRaisesRegex(
                ValueError, "mean reduction"
            ):
                train_one_epoch(
                    self.model, self.loader, nn.CrossEntropyLoss(reduction=reduction),
                    self.optimizer, "cpu",
                )
        for criterion in (
            lambda output, labels: output,
            lambda output, labels: output.sum() * float("nan"),
            None,
        ):
            with self.subTest(criterion=criterion), self.assertRaises(ValueError):
                train_one_epoch(
                    self.model, self.loader, criterion, self.optimizer, "cpu"
                )

    def test_invalid_model_output(self) -> None:
        for output in (
            torch.zeros(3),
            torch.zeros(3, 2),
            torch.full((2, 2), float("inf")),
        ):
            class BadModel(nn.Module):
                def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                    return output

            with self.subTest(shape=output.shape), self.assertRaisesRegex(
                ValueError, "batch 1"
            ):
                train_one_epoch(
                    BadModel(), self.loader, self.criterion, self.optimizer, "cpu"
                )

    def test_empty_after_a_valid_batch_is_not_ignored(self) -> None:
        batches = [
            (self.images[:2], self.labels[:2]),
            (self.images[:0], self.labels[:0]),
        ]
        with self.assertRaisesRegex(ValueError, "batch 2"):
            self.train(batches)

    def test_amp_disabled_by_default_and_on_cpu(self) -> None:
        for use_amp in (False, True):
            with self.subTest(use_amp=use_amp), mock.patch(
                "torch.amp.GradScaler", create=True
            ) as scaler_cls:
                metrics = train_one_epoch(
                    self.model, self.loader, self.criterion, self.optimizer,
                    "cpu", use_amp=use_amp,
                )
                scaler_cls.assert_not_called()
                self.assertEqual(metrics["num_samples"], 3)

    def test_amp_cpu_still_updates_parameters(self) -> None:
        optimizer = SGD(self.model.parameters(), lr=0.1)
        before = self.model.weight.detach().clone()
        train_one_epoch(
            self.model, self.loader, self.criterion, optimizer, "cpu", use_amp=True
        )
        self.assertFalse(torch.equal(before, self.model.weight))

    def test_use_amp_must_be_bool(self) -> None:
        with self.assertRaisesRegex(ValueError, "use_amp"):
            train_one_epoch(
                self.model, self.loader, self.criterion, self.optimizer,
                "cpu", use_amp="yes",
            )

    def test_amp_enabled_uses_scaler_sequence(self) -> None:
        scaler = mock.MagicMock()
        scaler.scale.side_effect = lambda loss: loss
        # Simulate CUDA on a CPU-only machine: the model and tensors stay on CPU.
        with mock.patch("train.torch.cuda.is_available", return_value=True), mock.patch(
            "train.torch.device", return_value=mock.Mock(type="cuda")
        ), mock.patch.object(
            torch.Tensor, "to", lambda tensor, *args, **kwargs: tensor
        ), mock.patch("train.torch.amp.autocast") as autocast:
            train_one_epoch(
                self.model, self.loader, self.criterion, self.optimizer,
                "cuda", use_amp=True, scaler=scaler,
            )
        self.assertEqual(scaler.scale.call_count, 2)
        self.assertEqual(scaler.step.call_count, 2)
        self.assertEqual(scaler.update.call_count, 2)
        self.assertTrue(all(c.kwargs["enabled"] for c in autocast.call_args_list))


if __name__ == "__main__":
    unittest.main()
