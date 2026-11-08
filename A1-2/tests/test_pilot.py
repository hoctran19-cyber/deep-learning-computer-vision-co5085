"""Kiểm thử lựa chọn subset pilot không cần ảnh Food-101."""

import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import torch
import yaml
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import ToTensor


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pilot
from dataset import create_dataloader


class PilotDatasetTests(unittest.TestCase):
    """Kiểm tra split, số lượng, membership và seed."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        (cls.root / "meta").mkdir()
        cls.classes = [f"class_{i:03d}" for i in range(101)]
        metadata = {name: [f"{name}/{j}" for j in range(750)] for name in cls.classes}
        (cls.root / "meta" / "train.json").write_text(json.dumps(metadata))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def dataset_factory(self, root, manifest, split, transform):
        self.assertIn(split, ("train", "validation"))
        classes = self.classes
        root = root.resolve()

        class FakeFoodDataset:
            def __init__(self) -> None:
                self.split = split
                self.classes = classes
                self.samples = [
                    {"class_name": name, "label": index,
                     "image_path": root / "images" / name / f"{number}.jpg"}
                    for index, name in enumerate(classes)
                    for number in (range(675) if split == "train" else range(675, 750))
                ]

            def __getitem__(self, index):
                return transform(Image.new("RGB", (4, 4))), self.samples[index]["label"]

            def __len__(self):
                return len(self.samples)

        return FakeFoodDataset()

    def test_balanced_reproducible_subset_and_dataloader(self) -> None:
        with patch.object(pilot, "Food101Dataset", side_effect=self.dataset_factory) as mock:
            first = pilot.prepare_pilot_datasets(self.root, self.root / "manifest.csv", ToTensor(), ToTensor())
            second = pilot.prepare_pilot_datasets(self.root, self.root / "manifest.csv", ToTensor(), ToTensor())
        self.assertEqual([call.args[2] for call in mock.call_args_list], ["train", "validation"] * 2)
        for selected, repeated, split, expected in zip(first, second, ("train", "validation"), (5, 2)):
            self.assertEqual(selected.split, split)
            self.assertEqual(selected.indices, repeated.indices)
            counts = Counter(selected.dataset.samples[index]["class_name"] for index in selected.indices)
            self.assertEqual(len(counts), 101)
            self.assertEqual(set(counts.values()), {expected})
            loader = create_dataloader(selected, batch_size=8, num_workers=0)
            images, labels = next(iter(loader))
            self.assertEqual(tuple(images.shape), (8, 3, 4, 4))
            self.assertEqual(tuple(labels.shape), (8,))

    def test_rejects_mismatched_metadata(self) -> None:
        with (self.root / "meta" / "train.json").open() as file:
            metadata = json.load(file)
        metadata[self.classes[0]][0] = f"{self.classes[0]}/unexpected"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "meta").mkdir()
            (root / "meta" / "train.json").write_text(json.dumps(metadata))
            with patch.object(pilot, "Food101Dataset", side_effect=self.dataset_factory):
                with self.assertRaisesRegex(ValueError, "membership does not match"):
                    pilot.prepare_pilot_datasets(root, root / "manifest.csv", ToTensor(), ToTensor())

    def test_rejects_incorrect_split_counts(self) -> None:
        def faulty(*args):
            dataset = self.dataset_factory(*args)
            dataset.samples.pop()
            return dataset

        with patch.object(pilot, "Food101Dataset", side_effect=faulty):
            with self.assertRaisesRegex(ValueError, "invalid per-class"):
                pilot.prepare_pilot_datasets(self.root, self.root / "manifest.csv", ToTensor(), ToTensor())

    def test_notebook_code_cells_compile(self) -> None:
        with (ROOT / "notebooks" / "A1_2_experiment.ipynb").open() as file:
            notebook = json.load(file)
        self.assertEqual(notebook["nbformat"], 4)
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                source = cell["source"]
                compile("".join(source) if isinstance(source, list) else source, f"cell {index}", "exec")
                self.assertEqual(cell["outputs"], [])

    def test_notebook_training_cell_runs_with_cpu_fixtures(self) -> None:
        import copy
        import time
        import numpy as np
        from src.fit import fit as actual_fit
        from src.optim import build_optimizer, build_scheduler
        from src.models import count_parameters

        with (ROOT / "notebooks" / "A1_2_experiment.ipynb").open() as file:
            notebook = json.load(file)
        cell = notebook["cells"][-1]
        source = cell["source"]
        source = "".join(source) if isinstance(source, list) else source
        with (ROOT / "configs" / "config.yaml").open() as file:
            config = yaml.safe_load(file)

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(nn.Flatten(), nn.Linear(12, 4), nn.ReLU())
                self.classifier = nn.Linear(4, 101)

            def forward(self, inputs):
                return self.classifier(self.features(inputs))

            def to(self, device):
                return super().to("cpu")

        def make_optimizer(model, config):
            return build_optimizer(model, config, classifier=model.classifier)

        def cpu_fit(model, train, validation, criterion, optimizer, device, config,
                    checkpoint, log, scheduler):
            return actual_fit(model, train, validation, criterion, optimizer, "cpu",
                              config, checkpoint, log, scheduler)

        train_data = TensorDataset(torch.randn(8, 3, 2, 2), torch.arange(8))
        val_data = TensorDataset(torch.randn(4, 3, 2, 2), torch.arange(4))
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            namespace = {
                "torch": torch, "random": __import__("random"), "np": np,
                "copy": copy, "time": time, "json": json, "base_config": config,
                "train_subset": train_data, "validation_subset": val_data,
                "run_dir": run_dir, "runtime_info": {"gpu": "synthetic"},
                "commit": "synthetic", "manifest_hash": "synthetic",
                "create_dataloader": lambda data, batch_size, **kwargs: DataLoader(data, batch_size=batch_size),
                "build_model": lambda *args, **kwargs: TinyModel(),
                "configure_finetuning": lambda *args: None,
                "count_parameters": count_parameters, "build_optimizer": make_optimizer,
                "build_scheduler": build_scheduler, "fit": cpu_fit,
            }
            with patch.object(torch.cuda, "synchronize"), \
                 patch.object(torch.cuda, "manual_seed_all"), \
                 patch.object(torch.cuda, "empty_cache"):
                exec(compile(source, "pilot notebook training", "exec"), namespace)
            for name in ("resnet50", "vit_b16"):
                self.assertTrue((run_dir / f"{name}_best.pt").is_file())
                summary = json.loads((run_dir / f"{name}_summary.json").read_text())
                self.assertEqual(summary["architecture"], name)
                self.assertEqual(len(summary["epoch_metrics"]), 2)


if __name__ == "__main__":
    unittest.main()
