"""Kiểm thử dừng sớm, lựa chọn mô hình và checkpoint."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import yaml
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, Subset, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import checkpoint
import fit
from early_stopping import EarlyStopping


class EarlyStoppingTests(unittest.TestCase):
    """Kiểm tra mốc cải thiện, hòa điểm và dừng khi hết patience."""

    def test_selection_and_patience_are_independent(self) -> None:
        stopping = EarlyStopping(patience=3, min_delta=0.1)
        self.assertTrue(stopping.update(1, 0.4, 0.5))
        self.assertTrue(stopping.update(2, 0.45, 0.4))
        self.assertTrue(stopping.update(3, 0.45, 0.8))
        self.assertEqual(stopping.bad_epochs, 2)
        self.assertEqual((stopping.best_epoch, stopping.best_score), (3, 0.45))
        self.assertTrue(stopping.update(4, 0.51, 0.2))
        self.assertEqual(stopping.bad_epochs, 0)
        self.assertFalse(stopping.should_stop)
        self.assertFalse(stopping.update(5, 0.5, 1.0))
        self.assertFalse(stopping.update(6, 0.51, 0.2))
        self.assertTrue(stopping.update(7, 0.51, 0.9))
        self.assertEqual(stopping.best_epoch, 7)
        self.assertTrue(stopping.should_stop)
        with self.assertRaisesRegex(RuntimeError, "already"):
            stopping.update(8, 0.9, 0.9)

    def test_strict_min_delta_boundary(self) -> None:
        stopping = EarlyStopping(patience=2, min_delta=0.125)
        stopping.update(1, 0.5, 0.5)
        self.assertTrue(stopping.update(2, 0.625, 0.4))
        self.assertEqual(stopping.bad_epochs, 1)
        self.assertTrue(stopping.update(3, 0.626, 0.3))
        self.assertEqual(stopping.bad_epochs, 0)

    def test_bad_configuration_and_metrics(self) -> None:
        for patience in (0, -1, True, 1.5):
            with self.subTest(patience=patience), self.assertRaises(ValueError):
                EarlyStopping(patience)
        for delta in (-1, float("nan"), float("inf"), True):
            with self.subTest(delta=delta), self.assertRaises(ValueError):
                EarlyStopping(2, delta)
        stopping = EarlyStopping(2)
        for score, accuracy in ((float("nan"), 0.5), (1.1, 0.5), (0.5, -1)):
            with self.subTest(score=score), self.assertRaises(ValueError):
                stopping.update(1, score, accuracy)
        stopping.update(1, 0.5, 0.5)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            stopping.update(1, 0.6, 0.6)


class FitTests(unittest.TestCase):
    """Kiểm thử workflow với validation giả lập và dữ liệu tổng hợp."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.checkpoint = self.root / "best.pt"
        self.log = self.root / "logs" / "epochs.jsonl"
        with (ROOT / "configs" / "config.yaml").open(encoding="utf-8") as file:
            self.config = yaml.safe_load(file)
        self.config["training"].update(
            max_epochs=6, early_stopping_patience=2, early_stopping_min_delta=0.0
        )
        self.model = nn.Linear(2, 2)
        self.optimizer = SGD(self.model.parameters(), lr=0.1)
        self.criterion = nn.CrossEntropyLoss()
        self.loader = DataLoader(
            TensorDataset(torch.randn(4, 2), torch.tensor([0, 1, 0, 1])),
            batch_size=2,
        )

    def run_fit(self, train_loader: object = None, val_loader: object = None) -> fit.FitResult:
        return fit.fit(
            self.model,
            self.loader if train_loader is None else train_loader,
            self.loader if val_loader is None else val_loader,
            self.criterion,
            self.optimizer,
            "cpu",
            self.config,
            self.checkpoint,
            self.log,
        )

    def test_ties_patience_and_checkpoint_preservation(self) -> None:
        scores = [(0.5, 0.5), (0.6, 0.5), (0.6, 0.8), (0.59, 0.99)]
        validation = [
            {"loss": 1.0, "accuracy": accuracy, "macro_f1": score, "num_samples": 4}
            for score, accuracy in scores
        ]
        def fake_train(*args: object) -> dict[str, float | int]:
            with torch.no_grad():
                self.model.weight.add_(1)
            return {"loss": 1.0, "accuracy": 0.5, "num_samples": 4}

        with patch.object(fit, "train_one_epoch", side_effect=fake_train), patch.object(
            fit, "evaluate_one_epoch", side_effect=validation
        ):
            result = self.run_fit()
        self.assertEqual(result["epochs_completed"], 4)
        self.assertTrue(result["stopped_early"])
        self.assertEqual(result["best_epoch"], 3)
        self.assertEqual(result["best_macro_f1"], 0.6)
        self.assertEqual(result["best_accuracy"], 0.8)
        self.assertEqual(result["best_checkpoint"], self.checkpoint)
        state = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        self.assertEqual(state["epoch"], 3)
        self.assertEqual(state["validation_metrics"]["accuracy"], 0.8)
        self.assertTrue(torch.allclose(state["model_state_dict"]["weight"] + 1, self.model.weight))
        records = [json.loads(line) for line in self.log.read_text().splitlines()]
        self.assertEqual([r["checkpoint_saved"] for r in records], [True, True, True, False])
        self.assertEqual(records[-1]["best_epoch"], 3)
        self.assertTrue(records[-1]["stopped_early"])

    def test_real_epoch_and_max_epochs(self) -> None:
        self.config["training"]["max_epochs"] = 1
        result = self.run_fit()
        self.assertFalse(result["stopped_early"])
        self.assertEqual(result["best_epoch"], 1)
        state = torch.load(self.checkpoint, weights_only=True)
        self.assertEqual(state["validation_metrics"]["num_samples"], 4)

    def test_rejects_test_split_even_through_subset(self) -> None:
        class TestDataset(TensorDataset):
            split = "test"

        data = TestDataset(torch.randn(4, 2), torch.tensor([0, 1, 0, 1]))
        val_loader = DataLoader(Subset(data, [0, 1]), batch_size=2)
        with self.assertRaisesRegex(ValueError, "invalid dataset split 'test'"):
            self.run_fit(val_loader=val_loader)
        self.assertFalse(self.checkpoint.exists())

    def test_scheduler_step_and_state(self) -> None:
        self.config["training"]["max_epochs"] = 1
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.5)
        fit.fit(self.model, self.loader, self.loader, self.criterion, self.optimizer,
                "cpu", self.config, self.checkpoint, self.log, scheduler)
        self.assertAlmostEqual(self.optimizer.param_groups[0]["lr"], 0.05)
        state = torch.load(self.checkpoint, weights_only=True)
        self.assertEqual(state["scheduler_state_dict"]["last_epoch"], 1)

    def test_atomic_save_preserves_previous_checkpoint_on_failure(self) -> None:
        checkpoint.save_checkpoint(self.checkpoint, {"epoch": 1})
        with patch.object(checkpoint.torch, "save", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                checkpoint.save_checkpoint(self.checkpoint, {"epoch": 2})
        self.assertEqual(torch.load(self.checkpoint, weights_only=True)["epoch"], 1)
        self.assertEqual(list(self.root.glob(".best.pt.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
