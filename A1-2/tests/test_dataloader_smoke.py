"""Integration tests for the DataLoader smoke-test CLI."""

import csv
import io
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import dataset
import test_dataloader as smoke
from PIL import Image
import torch
from torch.utils.data import DataLoader


class DataLoaderSmokeTests(unittest.TestCase):
    """Exercise real datasets and loaders with a small synthetic image fixture."""

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "meta").mkdir()
        classes = [f"class_{index:03d}" for index in reversed(range(101))]
        (self.root / "meta" / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
        self.manifest = self.root / "manifest.csv"
        counts = {"train": 3, "validation": 2, "test": 5}
        with self.manifest.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["image_path", "class_name", "label", "split"])
            writer.writeheader()
            for split, count in counts.items():
                for index in range(count):
                    class_name = classes[index]
                    relative = Path("images") / class_name / f"{split}_{index}.png"
                    (self.root / relative).parent.mkdir(parents=True, exist_ok=True)
                    Image.new("L", (5 + index, 7 + index), color=128).save(self.root / relative)
                    writer.writerow({"image_path": relative.as_posix(), "class_name": class_name, "label": index, "split": split})
        self.counts = counts

    def run_smoke(self, batch_size: int = 2) -> tuple[bool, str]:
        """Run smoke checks with small expected split sizes."""
        output = io.StringIO()
        with patch.object(smoke, "EXPECTED_COUNTS", self.counts), redirect_stdout(output), redirect_stderr(output):
            passed = smoke.run_smoke_tests(self.root, self.manifest, batch_size)
        return passed, output.getvalue()

    def test_real_loader_with_unsorted_classes_and_variable_image_sizes(self) -> None:
        passed, output = self.run_smoke()
        self.assertTrue(passed, output)
        self.assertIn("train: drop_last=False, full index coverage and last batch (1 samples)", output)
        self.assertIn("validation: drop_last=False, full index coverage and last batch (2 samples)", output)
        self.assertIn("test: drop_last=False, full index coverage and last batch (1 samples)", output)
        self.assertEqual(dataset.Food101Dataset(self.root, self.manifest, "train").class_to_idx["class_100"], 0)

    def test_batch_size_larger_than_split(self) -> None:
        passed, output = self.run_smoke(batch_size=8)
        self.assertTrue(passed, output)
        self.assertIn("last batch (3 samples)", output)

    def test_detects_dropped_last_batch(self) -> None:
        def broken_loader(data: dataset.Food101Dataset, batch_size: int, num_workers: int, pin_memory: bool) -> DataLoader:
            return DataLoader(data, batch_size=batch_size, shuffle=data.split == "train", drop_last=True, num_workers=num_workers, pin_memory=pin_memory)

        with patch.object(dataset, "create_dataloader", broken_loader):
            passed, output = self.run_smoke()
        self.assertFalse(passed)
        self.assertIn("drop_last must be False", output)

    def test_detects_wrong_shuffle(self) -> None:
        def broken_loader(data: dataset.Food101Dataset, batch_size: int, num_workers: int, pin_memory: bool) -> DataLoader:
            return DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

        with patch.object(dataset, "create_dataloader", broken_loader):
            passed, output = self.run_smoke()
        self.assertFalse(passed)
        self.assertIn("expected RandomSampler", output)

    def test_missing_image_is_reported(self) -> None:
        (self.root / "images" / "class_100" / "train_0.png").unlink()
        passed, output = self.run_smoke()
        self.assertFalse(passed)
        self.assertIn("Image not found", output)

    def test_batch_checks_reject_invalid_tensors(self) -> None:
        images = torch.zeros((2, 3, 5, 7), dtype=torch.float32)
        labels = torch.tensor([0, 100], dtype=torch.int64)
        smoke.check_batch((images, labels), 2)
        cases = [
            (torch.zeros((2, 1, 5, 7)), labels),
            (images.double(), labels),
            (images + 1.1, labels),
            (images + float("nan"), labels),
            (images, labels.float()),
            (images, torch.tensor([0, 101])),
            (images, torch.tensor([-1, 0])),
        ]
        for batch in cases:
            with self.subTest(batch=batch), self.assertRaises(smoke.SmokeTestError):
                smoke.check_batch(batch, 2)

    def test_cli_failure_has_nonzero_status_and_summary(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SRC / "test_dataloader.py"), "--dataset-root", str(self.root / "missing"), "--manifest", str(self.manifest)],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("TESTS FAILED", result.stdout)
        self.assertIn("Dataset root not found", result.stderr)


if __name__ == "__main__":
    unittest.main()
