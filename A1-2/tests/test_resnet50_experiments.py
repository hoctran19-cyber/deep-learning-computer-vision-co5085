"""Verify fine-tuning configuration and modes without running training."""

import copy
import csv
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch
import yaml
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import dataset as dataset_module
from src import fit as fit_module
from src import transforms as transforms_module
from src.experiments import resnet50_experiment_configs, resnet50_output_paths
from src.early_stopping import EarlyStopping
from src.models import build_resnet50, configure_finetuning, count_parameters
from src.optim import build_optimizer
from src.train import _new_grad_scaler, _set_training_mode


class ResNet50ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with (ROOT / "configs" / "config.yaml").open(encoding="utf-8") as file:
            cls.base = yaml.safe_load(file)
        cls.configs = resnet50_experiment_configs(cls.base)
        cls.model = build_resnet50(pretrained=False)

    def test_configurations_and_optimizer_parameter_sets(self) -> None:
        expected = {
            "resnet50_frozen": ("frozen_backbone", ("fc.",), 206949),
            "resnet50_partial": ("partial_finetuning", ("layer4.", "fc."), 15171685),
            "resnet50_full_finetune": ("full_finetuning", (), 23714981),
        }
        for name, (regime, prefixes, count) in expected.items():
            with self.subTest(name=name):
                config = self.configs[name]
                self.assertEqual(config["experiment"]["regime"], regime)
                self.assertEqual(config["training"]["regime"], regime)
                self.assertTrue(config["experiment"]["pretrained"])
                self.assertEqual(config["project"]["seed"], 42)
                training = config["training"]
                self.assertEqual(training["batch_size"], 32)
                self.assertEqual(training["max_epochs"], 30)
                self.assertEqual(training["early_stopping_patience"], 5)
                self.assertEqual(training["optimizer"], "AdamW")
                self.assertEqual(training["weight_decay"], 0.01)
                self.assertEqual(training["head_learning_rate"], 0.001)
                self.assertEqual(training["backbone_learning_rate"], 0.0001)
                self.assertEqual(training["scheduler"]["name"], "CosineAnnealingLR")
                self.assertEqual(training["scheduler"]["cosine_annealing"]["T_max"], 30)
                self.assertEqual(training["selection_metric"], "macro_f1")
                self.assertEqual(training["tie_break_metric"], "accuracy")
                self.assertTrue(training["use_amp"])
                configure_finetuning(self.model, regime)
                parameters = dict(self.model.named_parameters())
                actual = {name for name, p in parameters.items() if p.requires_grad}
                wanted = {
                    name for name in parameters
                    if not prefixes or name.startswith(prefixes)
                }
                self.assertEqual(actual, wanted)
                self.assertEqual(count_parameters(self.model)[1], count)
                optimizer = build_optimizer(self.model, config)
                included = [
                    id(p) for group in optimizer.param_groups for p in group["params"]
                ]
                self.assertEqual(set(included), {id(parameters[n]) for n in wanted})
                self.assertEqual(len(included), len(set(included)))
        self.assertEqual(self.base["training"]["regime"], "from_scratch")
        self.assertFalse(self.base["experiments"]["pretrained"])

    def test_independent_overrides_and_invalid_runs(self) -> None:
        base = copy.deepcopy(self.base)
        base["experiments"]["resnet50_runs"]["resnet50_frozen"]["training"] = {
            "batch_size": 16
        }
        configs = resnet50_experiment_configs(base)
        self.assertEqual(configs["resnet50_frozen"]["training"]["batch_size"], 16)
        self.assertEqual(configs["resnet50_partial"]["training"]["batch_size"], 32)
        configs["resnet50_frozen"]["training"]["scheduler"]["cosine_annealing"]["T_max"] = 1
        self.assertEqual(configs["resnet50_partial"]["training"]["scheduler"]["cosine_annealing"]["T_max"], 30)
        self.assertEqual(base["training"]["scheduler"]["cosine_annealing"]["T_max"], 30)
        base["experiments"]["resnet50_runs"]["resnet50_partial"]["pretrained"] = False
        with self.assertRaisesRegex(ValueError, "resnet50_partial"):
            resnet50_experiment_configs(base)

    def test_validation_selection_uses_macro_f1_then_accuracy(self) -> None:
        stopping = EarlyStopping(patience=5, min_delta=0.0)
        self.assertTrue(stopping.update(1, 0.6, 0.7))
        self.assertFalse(stopping.update(2, 0.5, 0.9))
        self.assertTrue(stopping.update(3, 0.6, 0.8))
        self.assertEqual(stopping.best_epoch, 3)
        self.assertEqual((stopping.best_score, stopping.best_accuracy), (0.6, 0.8))

    def test_scaler_factory_uses_available_amp_api(self) -> None:
        with patch.object(torch.amp, "GradScaler", create=True) as modern:
            self.assertIs(_new_grad_scaler(), modern.return_value)
            modern.assert_called_once_with("cuda")
        if not hasattr(torch.amp, "GradScaler"):
            with patch.object(torch.cuda.amp, "GradScaler") as legacy:
                self.assertIs(_new_grad_scaler(), legacy.return_value)
                legacy.assert_called_once_with()

    def test_fit_writes_selected_and_last_checkpoints_with_mock_epochs(self) -> None:
        config = copy.deepcopy(self.configs["resnet50_frozen"])
        config["training"].update(max_epochs=2, use_amp=False)
        model = nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters())
        train_metrics = {"loss": 1.0, "accuracy": 0.5, "num_samples": 2}
        validation = [
            {"loss": 1.0, "accuracy": 0.7, "macro_f1": 0.6, "num_samples": 2},
            {"loss": 1.0, "accuracy": 0.8, "macro_f1": 0.6, "num_samples": 2},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            paths = resnet50_output_paths(Path(tmp))["resnet50_frozen"]
            paths["directory"].mkdir()
            with patch.object(fit_module, "train_one_epoch", return_value=train_metrics), \
                 patch.object(fit_module, "evaluate_one_epoch", side_effect=validation):
                result = fit_module.fit(
                    model, [], [], nn.CrossEntropyLoss(), optimizer, "cpu", config,
                    paths["best_checkpoint"], paths["training_log"],
                    last_checkpoint_path=paths["last_checkpoint"],
                )
            self.assertEqual(result["best_epoch"], 2)
            best = torch.load(paths["best_checkpoint"], map_location="cpu", weights_only=True)
            last = torch.load(paths["last_checkpoint"], map_location="cpu", weights_only=False)
            self.assertEqual(best["epoch"], 2)
            self.assertEqual(best["validation_metrics"], validation[1])
            self.assertEqual(last["epoch"], 2)
            records = [json.loads(line) for line in paths["training_log"].read_text().splitlines()]
            self.assertEqual(len(records), 2)
            self.assertTrue(records[-1]["checkpoint_saved"])

    def test_isolated_paths_and_existing_directory_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = resnet50_output_paths(root / "outputs")
            all_files = []
            for name, run in paths.items():
                self.assertEqual(run["directory"], root / "outputs" / name)
                self.assertEqual(
                    {p.name for key, p in run.items() if key != "directory"},
                    {"best_checkpoint.pt", "last_checkpoint.pt", "training_log.jsonl",
                     "effective_config.yaml", "summary.json"},
                )
                all_files.extend(p for key, p in run.items() if key != "directory")
            self.assertEqual(len(all_files), len(set(all_files)))
            paths["resnet50_partial"]["directory"].mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                resnet50_output_paths(root / "outputs")
            self.assertFalse(paths["resnet50_frozen"]["directory"].exists())

    def test_frozen_batchnorm_buffers_and_head_mode(self) -> None:
        model = self.model
        batchnorms = {
            name: module for name, module in model.named_modules()
            if isinstance(module, nn.modules.batchnorm._BatchNorm)
        }
        self.assertTrue(batchnorms)
        for regime in ("frozen_backbone", "partial_finetuning", "full_finetuning", "from_scratch"):
            configure_finetuning(model, regime)
            model.eval()
            _set_training_mode(model)
            self.assertTrue(model.training)
            self.assertTrue(model.fc.training)
            for name, module in batchnorms.items():
                expected = regime in ("full_finetuning", "from_scratch") or (
                    regime == "partial_finetuning" and name.startswith("layer4.")
                )
                self.assertEqual(module.training, expected, (regime, name))
            for module in (model.bn1, model.layer3[0].bn1, model.layer4[0].bn1):
                before = (module.running_mean.clone(), module.running_var.clone(),
                          module.num_batches_tracked.clone())
                with torch.no_grad():
                    module(torch.randn(4, module.num_features, 4, 4))
                after = (module.running_mean, module.running_var, module.num_batches_tracked)
                self.assertEqual(not torch.equal(before[2], after[2]), module.training)
                if not module.training:
                    self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, after)))

    def test_full_dataset_loading_and_size_failures(self) -> None:
        class FakeDataset:
            def __init__(self, split, size):
                self.split = split
                self.size = size
                self.classes = ["shared"]

            def __len__(self):
                return self.size

        for sizes, failing_split in (
            ((68175, 7575), None), ((68174, 7575), "train"),
            ((68175, 7574), "validation"),
        ):
            datasets = [FakeDataset("train", sizes[0]), FakeDataset("validation", sizes[1])]
            with self.subTest(sizes=sizes), patch.object(
                dataset_module, "Food101Dataset", side_effect=datasets
            ) as factory:
                if failing_split:
                    with self.assertRaisesRegex(ValueError, f"{failing_split} dataset has"):
                        dataset_module.prepare_full_training_datasets("root", "manifest", "train_tf", "eval_tf")
                else:
                    result = dataset_module.prepare_full_training_datasets(
                        "root", "manifest", "train_tf", "eval_tf"
                    )
                    self.assertEqual(result, tuple(datasets))
                self.assertEqual([call.args[2] for call in factory.call_args_list], ["train", "validation"])
                self.assertTrue(all(call.kwargs == {"ignore_test_rows": True} for call in factory.call_args_list))

    def test_loader_skips_test_labels_and_image_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "meta").mkdir()
            (root / "meta" / "classes.txt").write_text(
                "\n".join(f"class_{i:03d}" for i in range(101))
            )
            (root / "images").mkdir()
            for name in ("train.jpg", "validation.jpg"):
                (root / "images" / name).touch()
            manifest = root / "manifest.csv"
            with manifest.open("w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(("image_path", "class_name", "label", "split"))
                writer.writerow(("images/train.jpg", "class_000", "0", "train"))
                writer.writerow(("images/validation.jpg", "class_000", "0", "validation"))
                writer.writerow(("images/test.jpg", "unknown", "not-an-int", "test"))
            for split in ("train", "validation"):
                data = dataset_module.Food101Dataset(root, manifest, split, ignore_test_rows=True)
                self.assertEqual(len(data), 1)
                self.assertEqual(data.samples[0]["image_path"].name, f"{split}.jpg")
            with self.assertRaisesRegex(ValueError, "Unknown class"):
                dataset_module.Food101Dataset(root, manifest, "train")
            with self.assertRaisesRegex(ValueError, "cannot ignore test"):
                dataset_module.Food101Dataset(root, manifest, "test", ignore_test_rows=True)

    def test_notebook_cells_parse_and_use_isolated_paths(self) -> None:
        notebook = json.loads((ROOT / "notebooks" / "A1_2_experiment.ipynb").read_text())
        code = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"] if cell["cell_type"] == "code"
        )
        compile(code, "A1_2_experiment.ipynb", "exec")
        self.assertIn("resnet50_experiment_configs(base_config)", code)
        self.assertIn("resnet50_output_paths(OUTPUT_BASE)", code)
        self.assertIn("last_checkpoint_path=paths['last_checkpoint']", code)
        self.assertNotIn("build_model(architecture", code)
        self.assertIn("prepare_full_training_datasets(", code)
        self.assertNotIn("prepare_pilot_datasets", code)
        self.assertNotIn("train_subset", code)
        self.assertNotIn("validation_subset", code)

    def test_notebook_prepares_full_splits_without_test_dataset(self) -> None:
        notebook = json.loads((ROOT / "notebooks" / "A1_2_experiment.ipynb").read_text())
        cell = next(cell for cell in notebook["cells"] if cell.get("id") == "db6cb6ee")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.csv"
            manifest.write_bytes(b"manifest")
            train = range(68175)
            validation = range(7575)
            namespace = {
                "REPO_DIR": ROOT.parent, "DATASET_ROOT": root / "dataset",
                "MANIFEST": manifest, "OUTPUT_BASE": root / "outputs",
            }
            with patch.object(dataset_module, "prepare_full_training_datasets",
                              return_value=(train, validation)) as prepare, \
                 patch.object(transforms_module, "build_train_transform", return_value="train_tf"), \
                 patch.object(transforms_module, "build_eval_transform", return_value="eval_tf"):
                exec(compile("".join(cell["source"]), "notebook-setup", "exec"), namespace)
            prepare.assert_called_once_with(root / "dataset", manifest, "train_tf", "eval_tf")
            self.assertIs(namespace["train_dataset"], train)
            self.assertIs(namespace["validation_dataset"], validation)
            self.assertEqual(len(namespace["experiment_configs"]), 3)

    def test_notebook_wires_each_run_without_training(self) -> None:
        notebook = json.loads((ROOT / "notebooks" / "A1_2_experiment.ipynb").read_text())
        cell = next(cell for cell in notebook["cells"] if cell.get("id") == "997c7cc0")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            configs = resnet50_experiment_configs(self.base)
            paths = resnet50_output_paths(root / "outputs")
            model = Mock()
            model.to.return_value = model
            build_model = Mock(return_value=model)
            configure = Mock()
            fit_calls = []

            def fake_fit(model, train, validation, criterion, optimizer, device,
                         config, checkpoint, log, scheduler, *, last_checkpoint_path):
                self.assertIs(train, namespace["train_dataset"])
                self.assertIs(validation, namespace["validation_dataset"])
                fit_calls.append((copy.deepcopy(config), checkpoint, log, last_checkpoint_path))
                checkpoint.write_text("mock checkpoint")
                last_checkpoint_path.write_text("mock checkpoint")
                log.write_text('{"epoch": 1}\n')
                return {"best_epoch": 1, "best_macro_f1": 0.5, "best_accuracy": 0.6,
                        "epochs_completed": 1, "stopped_early": False}

            namespace = {
                "experiment_configs": configs, "output_paths": paths,
                "OUTPUT_BASE": root / "outputs", "DATASET_ROOT": root / "dataset",
                "MANIFEST": root / "manifest.csv", "train_dataset": range(68175),
                "validation_dataset": range(7575), "runtime_info": {}, "commit": "commit",
                "manifest_hash": "hash", "torch": torch, "random": __import__("random"),
                "np": __import__("numpy"), "time": time, "json": json, "yaml": yaml,
                "create_dataloader": Mock(side_effect=lambda dataset, **kwargs: dataset),
                "build_model": build_model, "configure_finetuning": configure,
                "count_parameters": Mock(return_value=(10, 5)),
                "build_optimizer": Mock(), "build_scheduler": Mock(), "fit": fake_fit,
            }
            with patch.object(torch.cuda, "synchronize"), \
                 patch.object(torch.cuda, "manual_seed_all"), \
                 patch.object(torch.cuda, "empty_cache"):
                exec(compile("".join(cell["source"]), "notebook-cell", "exec"), namespace)
            self.assertEqual(len(fit_calls), 3)
            for name, (config, checkpoint, log, last) in zip(paths, fit_calls):
                self.assertEqual(config["experiment"]["name"], name)
                self.assertEqual(config["training"]["regime"], config["experiment"]["regime"])
                self.assertFalse(config["evaluation"]["evaluate_test_after_validation_selection"])
                self.assertEqual((checkpoint, log, last), (
                    paths[name]["best_checkpoint"], paths[name]["training_log"],
                    paths[name]["last_checkpoint"],
                ))
                self.assertTrue(all(path.is_file() for key, path in paths[name].items()
                                    if key != "directory"))
                with paths[name]["effective_config"].open() as file:
                    self.assertEqual(yaml.safe_load(file), config)
                with paths[name]["summary"].open() as file:
                    summary = json.load(file)
                self.assertEqual(summary["experiment"], name)
                self.assertEqual(summary["dataset_name"], "food101")
                self.assertEqual(summary["manifest_path"], str(root / "manifest.csv"))
                self.assertEqual(summary["manifest_sha256"], "hash")
                self.assertEqual((summary["train_samples"], summary["validation_samples"]), (68175, 7575))
                self.assertEqual(summary["epochs_completed"], 1)
                self.assertEqual(summary["source_commit"], "commit")
                self.assertEqual(summary["runtime"], {})
                self.assertEqual(summary["seed"], 42)
                self.assertTrue(summary["pretrained"])
                self.assertEqual(summary["batch_size"], 32)
                self.assertEqual(summary["best_epoch"], 1)
                self.assertEqual((summary["validation_macro_f1"], summary["validation_accuracy"]), (0.5, 0.6))
                self.assertEqual((summary["total_parameters"], summary["trainable_parameters"]), (10, 5))
                self.assertIn("fit_wall_seconds_including_validation_and_io", summary)
                self.assertEqual(summary["config"]["training"], config["training"])
            self.assertEqual(
                [call.kwargs["pretrained"] for call in build_model.call_args_list],
                [True, True, True],
            )
            self.assertEqual(
                [call.args[1] for call in configure.call_args_list],
                ["frozen_backbone", "partial_finetuning", "full_finetuning"],
            )


if __name__ == "__main__":
    unittest.main()
