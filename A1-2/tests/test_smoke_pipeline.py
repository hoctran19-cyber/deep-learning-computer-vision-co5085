"""Kiểm thử tích hợp pipeline với CNN và dữ liệu ảnh giả lập."""

import copy
import json
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import yaml
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import fit
import optim
import smoke_test


class SyntheticPipelineTests(unittest.TestCase):
    """Kiểm tra chạy thật, artifact, tính lặp lại và lỗi đầu vào."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        with (ROOT / "configs" / "config.yaml").open(encoding="utf-8") as file:
            self.config = yaml.safe_load(file)

    def test_end_to_end_and_reproducibility(self) -> None:
        snapshots = []
        for index in range(2):
            output = self.root / str(index)
            with patch.object(fit, "train_one_epoch", wraps=fit.train_one_epoch) as train, \
                 patch.object(fit, "evaluate_one_epoch", wraps=fit.evaluate_one_epoch) as evaluate, \
                 patch.object(socket.socket, "connect", side_effect=AssertionError("network access")):
                result = smoke_test.run_smoke_test(self.config, output)
            self.assertTrue(result["stopped_early"])
            self.assertEqual(result["epochs_completed"], 2)
            self.assertEqual(train.call_count, 2)
            self.assertEqual(evaluate.call_count, 2)
            checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
            self.assertEqual(checkpoint["epoch"], result["best_epoch"])
            for key in ("model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "validation_metrics"):
                self.assertIsInstance(checkpoint[key], dict)
            self.assertTrue(checkpoint["optimizer_state_dict"]["state"])
            self.assertEqual(checkpoint["scheduler_state_dict"]["last_epoch"], result["best_epoch"])
            records = [json.loads(line) for line in (output / "epochs.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 2)
            self.assertTrue(records[-1]["stopped_early"])
            self.assertEqual(checkpoint["validation_metrics"], records[result["best_epoch"] - 1]["validation"])
            snapshots.append((result, records, checkpoint))
        first, second = snapshots
        self.assertEqual(first[1], second[1])
        for key in first[2]["model_state_dict"]:
            self.assertTrue(torch.equal(first[2]["model_state_dict"][key], second[2]["model_state_dict"][key]))

    def test_invalid_configuration_and_data(self) -> None:
        config = copy.deepcopy(self.config)
        config["training"]["optimizer"] = "invalid"
        with self.assertRaisesRegex(ValueError, "Unsupported optimizer"):
            smoke_test.run_smoke_test(config, self.root / "bad-optimizer")
        config = copy.deepcopy(self.config)
        config["training"]["scheduler"]["step"]["step_size"] = 0
        config["training"]["scheduler"]["name"] = "StepLR"
        with self.assertRaisesRegex(ValueError, "step_size"):
            smoke_test.run_smoke_test(config, self.root / "bad-scheduler")
        with patch.object(
            torch, "randint",
            side_effect=[torch.full((12,), 9), torch.zeros(7, dtype=torch.long)],
        ):
            with self.assertRaisesRegex(ValueError, "label outside class range"):
                smoke_test.run_smoke_test(self.config, self.root / "bad-data")

    def test_classifier_validation_and_no_overwrite(self) -> None:
        model = smoke_test.SmallCNN()
        with self.assertRaisesRegex(ValueError, "Unsupported model type"):
            optim.build_optimizer(model, self.config)
        with self.assertRaisesRegex(ValueError, "child module"):
            optim.build_optimizer(model, self.config, classifier=nn.Linear(4, 3))
        with self.assertRaisesRegex(ValueError, "child module"):
            optim.build_optimizer(model, self.config, classifier=model)
        config = copy.deepcopy(self.config)
        config["training"]["regime"] = "full_finetuning"
        optimizer = optim.build_optimizer(model, config, classifier=model.classifier)
        self.assertEqual(
            [group["lr"] for group in optimizer.param_groups],
            [config["training"]["backbone_learning_rate"], config["training"]["head_learning_rate"]],
        )
        self.assertEqual(
            {id(p) for p in optimizer.param_groups[1]["params"]},
            {id(p) for p in model.classifier.parameters()},
        )
        output = self.root / "existing"
        output.mkdir()
        (output / "best.pt").write_text("do not overwrite")
        with self.assertRaises(FileExistsError):
            smoke_test.run_smoke_test(self.config, output)
        self.assertEqual((output / "best.pt").read_text(), "do not overwrite")

    def test_cli_and_existing_output_exit_nonzero(self) -> None:
        output = self.root / "cli"
        command = [
            sys.executable, str(ROOT / "src" / "smoke_test.py"),
            "--output-dir", str(output),
        ]
        first = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertIn("SMOKE TEST PASSED", first.stdout)
        second = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertNotEqual(second.returncode, 0)
        self.assertIn("SMOKE TEST FAILED", second.stderr)


if __name__ == "__main__":
    unittest.main()
