"""Verify fine-tuning configuration and modes without running training."""

import copy
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

from src import fit as fit_module
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
        configure_finetuning(model, "frozen_backbone")
        model.eval()
        _set_training_mode(model)
        self.assertTrue(model.training)
        self.assertTrue(model.fc.training)
        batchnorms = [
            module for module in model.modules()
            if isinstance(module, nn.modules.batchnorm._BatchNorm)
        ]
        self.assertTrue(batchnorms)
        self.assertTrue(all(not module.training for module in batchnorms))
        before = (model.bn1.running_mean.clone(), model.bn1.running_var.clone(),
                  model.bn1.num_batches_tracked.clone())
        with torch.no_grad():
            model.bn1(torch.randn(4, 64, 4, 4))
        for old, new in zip(before, (model.bn1.running_mean, model.bn1.running_var,
                                     model.bn1.num_batches_tracked)):
            self.assertTrue(torch.equal(old, new))
        for regime in ("partial_finetuning", "full_finetuning", "from_scratch"):
            configure_finetuning(model, regime)
            _set_training_mode(model)
            self.assertTrue(all(module.training for module in batchnorms))
            self.assertTrue(model.fc.training)

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
                fit_calls.append((copy.deepcopy(config), checkpoint, log, last_checkpoint_path))
                checkpoint.write_text("mock checkpoint")
                last_checkpoint_path.write_text("mock checkpoint")
                log.write_text('{"epoch": 1}\n')
                return {"best_epoch": 1, "best_macro_f1": 0.5, "best_accuracy": 0.6}

            namespace = {
                "experiment_configs": configs, "output_paths": paths,
                "OUTPUT_BASE": root / "outputs", "DATASET_ROOT": root / "dataset",
                "MANIFEST": root / "manifest.csv", "train_subset": [1, 2],
                "validation_subset": [3], "runtime_info": {}, "commit": "commit",
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
                    self.assertEqual(json.load(file)["experiment"], name)
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
