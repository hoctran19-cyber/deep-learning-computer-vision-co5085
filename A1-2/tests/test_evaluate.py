"""Kiểm thử vòng validation và các chỉ số trên dữ liệu giả lập."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from evaluate import evaluate_one_epoch


class ValidationEpochTests(unittest.TestCase):
    """Kiểm tra metric, chế độ mô hình và các đầu vào không hợp lệ."""

    def setUp(self) -> None:
        self.images = torch.tensor([
            [4.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
        ])
        self.labels = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
        self.loader = DataLoader(
            TensorDataset(self.images, self.labels), batch_size=2, shuffle=False
        )
        self.model = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.model.weight.copy_(torch.eye(4))
        self.criterion = nn.CrossEntropyLoss()

    def evaluate(self, batches: object) -> dict[str, float | int]:
        """Chạy validation trên CPU."""
        return evaluate_one_epoch(self.model, batches, self.criterion, "cpu")

    def test_metrics_and_unequal_final_batch(self) -> None:
        expected_loss = self.criterion(self.images, self.labels).item()
        metrics = self.evaluate(self.loader)
        self.assertEqual(metrics["num_samples"], 5)
        self.assertAlmostEqual(metrics["loss"], expected_loss, places=6)
        self.assertAlmostEqual(metrics["accuracy"], 2 / 5)
        # F1: 0.4, 0.5, 0 (dự đoán nhưng không có nhãn), 0 (vắng mặt).
        self.assertAlmostEqual(metrics["macro_f1"], (0.4 + 0.5) / 4)

    def test_no_parameter_updates_or_gradients(self) -> None:
        self.model.weight.grad = torch.ones_like(self.model.weight)
        before = self.model.weight.detach().clone()
        grad_before = self.model.weight.grad.clone()
        with patch.object(self.model, "forward", wraps=self.model.forward) as forward:
            metrics = self.evaluate(self.loader)
        self.assertEqual(metrics["num_samples"], 5)
        self.assertEqual(forward.call_count, 3)
        self.assertTrue(torch.equal(before, self.model.weight))
        self.assertTrue(torch.equal(grad_before, self.model.weight.grad))

        class NoGradModel(nn.Module):
            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                if torch.is_grad_enabled() or not torch.is_inference_mode_enabled():
                    raise AssertionError("inference mode is not active")
                return inputs

        evaluate_one_epoch(NoGradModel(), self.loader, self.criterion, "cpu")

    def test_mode_restoration_including_mixed_child_modes(self) -> None:
        model = nn.Sequential(nn.Linear(4, 4), nn.Dropout())
        for training in (True, False):
            model.train(training)
            model[0].train(not training)
            before = [module.training for module in model.modules()]
            evaluate_one_epoch(model, self.loader, self.criterion, "cpu")
            self.assertEqual(before, [module.training for module in model.modules()])
            with self.assertRaisesRegex(ValueError, "batch 1"):
                evaluate_one_epoch(model, [("bad", self.labels)], self.criterion, "cpu")
            self.assertEqual(before, [module.training for module in model.modules()])

    def test_mode_restored_when_forward_raises(self) -> None:
        class FailingModel(nn.Module):
            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                raise RuntimeError("forward failed")

        model = FailingModel()
        model.train()
        with self.assertRaisesRegex(RuntimeError, "forward failed"):
            evaluate_one_epoch(model, self.loader, self.criterion, "cpu")
        self.assertTrue(model.training)

    def test_empty_and_noniterable_dataloader(self) -> None:
        for batches in ([], None):
            with self.subTest(batches=batches), self.assertRaisesRegex(
                ValueError, "dataloader"
            ):
                self.evaluate(batches)

    def test_malformed_batches(self) -> None:
        cases = [
            "bad",
            (self.images,),
            (self.images, self.labels, self.labels),
            ([1, 2], self.labels),
            (torch.tensor(1.0), self.labels),
            (self.images, self.labels.unsqueeze(1)),
            (self.images, self.labels[:2]),
            (torch.empty((0, 4)), torch.empty(0, dtype=torch.long)),
            (self.images, self.labels.float()),
            (self.images, torch.tensor([0, 1, 2, 3, 4])),
            (self.images, torch.tensor([0, 1, 2, 3, -1])),
        ]
        for batch in cases:
            with self.subTest(batch=repr(batch)), self.assertRaisesRegex(
                ValueError, "batch 1"
            ):
                self.evaluate([batch])

    def test_invalid_criterion(self) -> None:
        for criterion in (
            nn.CrossEntropyLoss(reduction="sum"),
            nn.CrossEntropyLoss(reduction="none"),
            lambda output, labels: output,
            lambda output, labels: output.sum() * float("nan"),
            None,
        ):
            with self.subTest(criterion=criterion), self.assertRaises(ValueError):
                evaluate_one_epoch(self.model, self.loader, criterion, "cpu")

    def test_invalid_logits(self) -> None:
        for output in (
            torch.zeros(2),
            torch.zeros(3, 4),
            torch.empty(2, 0),
            torch.full((2, 4), float("inf")),
            torch.full((2, 4), float("nan")),
            "not a tensor",
        ):
            class BadModel(nn.Module):
                def forward(self, inputs: torch.Tensor) -> object:
                    return output

            with self.subTest(output=repr(output)), self.assertRaisesRegex(
                ValueError, "batch 1"
            ):
                evaluate_one_epoch(BadModel(), self.loader, self.criterion, "cpu")

    def test_number_of_classes_must_remain_constant(self) -> None:
        class ChangingClasses(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.calls = 0

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                self.calls += 1
                return torch.zeros(inputs.shape[0], 4 if self.calls == 1 else 3)

        with self.assertRaisesRegex(ValueError, "number of classes changed"):
            evaluate_one_epoch(ChangingClasses(), self.loader, self.criterion, "cpu")


if __name__ == "__main__":
    unittest.main()
