"""Kiểm thử cấu hình optimizer và scheduler."""

import copy
import math
import sys
import unittest
from pathlib import Path

import torch
import yaml
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import models
import optim


class OptimizerTests(unittest.TestCase):
    """Kiểm tra các chế độ và optimizer với cả hai kiến trúc thực tế."""

    @classmethod
    def setUpClass(cls) -> None:
        with (ROOT / "configs" / "config.yaml").open(encoding="utf-8") as file:
            cls.config = yaml.safe_load(file)

    def test_parameter_groups_for_architectures_and_regimes(self) -> None:
        for architecture in ("resnet50", "vit_b16"):
            model = models.build_model(architecture, pretrained=False)
            head = model.fc if architecture == "resnet50" else model.heads
            head_ids = {id(parameter) for parameter in head.parameters()}
            for regime in self.config["experiments"]["regimes"]:
                models.configure_finetuning(model, regime)
                expected = {
                    id(parameter) for parameter in model.parameters()
                    if parameter.requires_grad
                }
                for name, optimizer_type in (("AdamW", AdamW), ("SGD", SGD)):
                    with self.subTest(model=architecture, regime=regime, optimizer=name):
                        config = copy.deepcopy(self.config)
                        config["training"].update(optimizer=name, regime=regime)
                        optimizer = optim.build_optimizer(model, config)
                        self.assertIsInstance(optimizer, optimizer_type)
                        found: list[int] = []
                        for group in optimizer.param_groups:
                            ids = {id(p) for p in group["params"]}
                            self.assertTrue(ids)
                            self.assertTrue(ids <= head_ids or ids.isdisjoint(head_ids))
                            if regime == "from_scratch":
                                lr = config["training"]["scratch_learning_rate"]
                            elif ids <= head_ids:
                                lr = config["training"]["head_learning_rate"]
                            else:
                                lr = config["training"]["backbone_learning_rate"]
                            self.assertEqual(group["lr"], lr)
                            self.assertEqual(group["weight_decay"], config["training"]["weight_decay"])
                            if name == "SGD":
                                self.assertEqual(group["momentum"], config["training"]["momentum"])
                            found.extend(id(p) for p in group["params"])
                        self.assertEqual(set(found), expected)
                        self.assertEqual(len(found), len(set(found)))
                        self.assertEqual(len(optimizer.param_groups), 1 if regime == "frozen_backbone" else 2)

    def test_head_frozen_and_no_trainable_parameters(self) -> None:
        model = models.build_resnet50()
        models.freeze_parameters(model.fc)
        optimizer = optim.build_optimizer(model, self.config)
        actual = {id(p) for group in optimizer.param_groups for p in group["params"]}
        self.assertTrue(actual)
        self.assertTrue(actual.isdisjoint({id(p) for p in model.fc.parameters()}))
        models.freeze_parameters(model)
        with self.assertRaisesRegex(ValueError, "no trainable parameters"):
            optim.build_optimizer(model, self.config)

    def test_optimizer_step_updates_only_head_in_frozen_mode(self) -> None:
        for name in ("AdamW", "SGD"):
            with self.subTest(optimizer=name):
                model = models.build_resnet50()
                models.configure_finetuning(model, "frozen_backbone")
                config = copy.deepcopy(self.config)
                config["training"].update(
                    optimizer=name, regime="frozen_backbone"
                )
                optimizer = optim.build_optimizer(model, config)
                head_before = model.fc.weight.detach().clone()
                backbone_before = model.conv1.weight.detach().clone()
                model.fc.weight.grad = torch.ones_like(model.fc.weight)
                optimizer.step()
                self.assertFalse(torch.equal(head_before, model.fc.weight))
                self.assertTrue(torch.equal(backbone_before, model.conv1.weight))

    def test_invalid_optimizer_configuration(self) -> None:
        model = models.build_resnet50()
        for key, value in (
            ("optimizer", "RMSprop"), ("regime", "unknown"),
            ("weight_decay", -1), ("weight_decay", float("nan")),
            ("scratch_learning_rate", 0), ("scratch_learning_rate", float("inf")),
            ("scratch_learning_rate", True), ("scratch_learning_rate", "0.001"),
        ):
            config = copy.deepcopy(self.config)
            config["training"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                optim.build_optimizer(model, config)
        for value in (-1, 1, float("nan"), True):
            config = copy.deepcopy(self.config)
            config["training"].update(optimizer="SGD", momentum=value)
            with self.subTest(momentum=value), self.assertRaisesRegex(ValueError, "momentum"):
                optim.build_optimizer(model, config)
        config = copy.deepcopy(self.config)
        config["training"].update(regime="full_finetuning", backbone_learning_rate=-1)
        with self.assertRaisesRegex(ValueError, "learning rates"):
            optim.build_optimizer(model, config)
        with self.assertRaisesRegex(ValueError, "Unsupported model type"):
            optim.build_optimizer(torch.nn.Linear(2, 2), self.config)
        with self.assertRaisesRegex(ValueError, "training"):
            optim.build_optimizer(model, {})


class SchedulerTests(unittest.TestCase):
    """Kiểm tra cấu hình, bước scheduler và siêu tham số không hợp lệ."""

    def setUp(self) -> None:
        with (ROOT / "configs" / "config.yaml").open(encoding="utf-8") as file:
            self.config = yaml.safe_load(file)
        model = models.build_resnet50()
        models.configure_finetuning(model, "full_finetuning")
        self.config["training"]["regime"] = "full_finetuning"
        self.optimizer = optim.build_optimizer(model, self.config)

    def test_cosine_annealing_step_updates_both_rates(self) -> None:
        options = self.config["training"]["scheduler"]["cosine_annealing"]
        options.update(T_max=4, eta_min=0.0)
        scheduler = optim.build_scheduler(self.optimizer, self.config)
        self.assertIsInstance(scheduler, CosineAnnealingLR)
        initial = [group["lr"] for group in self.optimizer.param_groups]
        self.optimizer.step()
        scheduler.step()
        for before, group in zip(initial, self.optimizer.param_groups):
            self.assertAlmostEqual(group["lr"], before * (1 + math.cos(math.pi / 4)) / 2)

    def test_step_lr_updates_both_rates(self) -> None:
        scheduler_config = self.config["training"]["scheduler"]
        scheduler_config["name"] = "StepLR"
        scheduler_config["step"].update(step_size=2, gamma=0.5)
        scheduler = optim.build_scheduler(self.optimizer, self.config)
        self.assertIsInstance(scheduler, StepLR)
        initial = [group["lr"] for group in self.optimizer.param_groups]
        for _ in range(2):
            self.optimizer.step()
            scheduler.step()
        for before, group in zip(initial, self.optimizer.param_groups):
            self.assertAlmostEqual(group["lr"], before * 0.5)

    def test_invalid_scheduler_configuration(self) -> None:
        cases = (
            ("CosineAnnealingLR", "cosine_annealing", "T_max", 0),
            ("CosineAnnealingLR", "cosine_annealing", "T_max", 1.5),
            ("CosineAnnealingLR", "cosine_annealing", "T_max", True),
            ("CosineAnnealingLR", "cosine_annealing", "eta_min", -1),
            ("CosineAnnealingLR", "cosine_annealing", "eta_min", 0.01),
            ("CosineAnnealingLR", "cosine_annealing", "eta_min", float("nan")),
            ("StepLR", "step", "step_size", 0),
            ("StepLR", "step", "gamma", 0),
            ("StepLR", "step", "gamma", 1.1),
            ("StepLR", "step", "gamma", float("inf")),
        )
        for name, section, key, value in cases:
            config = copy.deepcopy(self.config)
            scheduler = config["training"]["scheduler"]
            scheduler["name"] = name
            scheduler[section][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                optim.build_scheduler(self.optimizer, config)
        config = copy.deepcopy(self.config)
        config["training"]["scheduler"]["name"] = "unknown"
        with self.assertRaisesRegex(ValueError, "Unsupported scheduler"):
            optim.build_scheduler(self.optimizer, config)
        with self.assertRaisesRegex(ValueError, "scheduler"):
            optim.build_scheduler(self.optimizer, {"training": {}})


if __name__ == "__main__":
    unittest.main()
