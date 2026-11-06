"""Unit tests for Food-101 model builders."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import yaml
from torch import nn
from torchvision.models import ResNet50_Weights, ViT_B_16_Weights
from torchvision.models.resnet import ResNet
from torchvision.models.vision_transformer import VisionTransformer


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import models


class ResNet50BuilderTests(unittest.TestCase):
    """Verify ResNet-50 construction without downloading weights."""

    def test_random_initialization_head_and_forward(self) -> None:
        model = models.build_resnet50(pretrained=False)
        self.assertIsInstance(model, ResNet)
        self.assertIsInstance(model.fc, nn.Linear)
        self.assertEqual(model.fc.in_features, 2048)
        self.assertEqual(model.fc.out_features, 101)
        model.eval()
        with torch.no_grad():
            output = model(torch.randn(2, 3, 64, 64))
        self.assertEqual(tuple(output.shape), (2, 101))
        self.assertTrue(torch.isfinite(output).all().item())

    def test_configurable_class_count(self) -> None:
        model = models.build_resnet50(num_classes=7, pretrained=False)
        self.assertEqual(model.fc.out_features, 7)
        model.eval()
        with torch.no_grad():
            output = model(torch.randn(1, 3, 64, 64))
        self.assertEqual(tuple(output.shape), (1, 7))

    def test_pretrained_weight_selection_without_download(self) -> None:
        actual_resnet50 = models.resnet50
        with patch.object(models, "resnet50") as mocked:
            mocked.side_effect = lambda *, weights: actual_resnet50(weights=None)
            model = models.build_resnet50(pretrained=True)
            mocked.assert_called_once_with(weights=ResNet50_Weights.DEFAULT)
        self.assertEqual(model.fc.out_features, 101)

    def test_random_weight_selection(self) -> None:
        actual_resnet50 = models.resnet50
        with patch.object(models, "resnet50") as mocked:
            mocked.side_effect = actual_resnet50
            models.build_resnet50(pretrained=False)
            mocked.assert_called_once_with(weights=None)

    def test_rejects_nonpositive_class_count(self) -> None:
        for count in (0, -1):
            with self.subTest(count=count), self.assertRaises(ValueError):
                models.build_resnet50(num_classes=count)


class ViTB16BuilderTests(unittest.TestCase):
    """Verify ViT-B/16 construction without downloading weights."""

    def test_random_initialization_head_and_forward(self) -> None:
        actual_vit_b16 = models.vit_b_16
        with patch.object(models, "vit_b_16", wraps=actual_vit_b16) as mocked:
            model = models.build_vit_b16(pretrained=False)
            mocked.assert_called_once_with(weights=None)
        self.assertIsInstance(model, VisionTransformer)
        self.assertIsInstance(model.heads.head, nn.Linear)
        self.assertEqual(model.heads.head.in_features, 768)
        self.assertEqual(model.heads.head.out_features, 101)
        model.eval()
        with torch.no_grad():
            output = model(torch.randn(2, 3, 224, 224))
        self.assertEqual(tuple(output.shape), (2, 101))
        self.assertTrue(torch.isfinite(output).all().item())

    def test_configurable_class_count(self) -> None:
        model = models.build_vit_b16(num_classes=7, pretrained=False)
        self.assertEqual(model.heads.head.out_features, 7)
        model.eval()
        with torch.no_grad():
            output = model(torch.randn(1, 3, 224, 224))
        self.assertEqual(tuple(output.shape), (1, 7))

    def test_pretrained_weight_selection_without_download(self) -> None:
        fake_model = SimpleNamespace(
            heads=SimpleNamespace(head=nn.Linear(768, 1000))
        )
        with patch.object(models, "vit_b_16", return_value=fake_model) as mocked:
            model = models.build_vit_b16(num_classes=7, pretrained=True)
            mocked.assert_called_once_with(weights=ViT_B_16_Weights.DEFAULT)
        self.assertIs(model, fake_model)
        self.assertEqual(model.heads.head.in_features, 768)
        self.assertEqual(model.heads.head.out_features, 7)

    def test_rejects_nonpositive_class_count(self) -> None:
        with patch.object(models, "vit_b_16") as mocked:
            for count in (0, -1):
                with self.subTest(count=count), self.assertRaises(ValueError):
                    models.build_vit_b16(num_classes=count)
            mocked.assert_not_called()


class ModelFactoryTests(unittest.TestCase):
    """Verify architecture selection and argument forwarding."""

    def test_selects_both_builders_and_forwards_options(self) -> None:
        for name, builder in (
            ("resnet50", "build_resnet50"),
            ("vit_b16", "build_vit_b16"),
        ):
            for pretrained in (False, True):
                with self.subTest(name=name, pretrained=pretrained):
                    with patch.object(models, builder) as mocked:
                        result = models.build_model(
                            name, num_classes=7, pretrained=pretrained
                        )
                        mocked.assert_called_once_with(
                            num_classes=7, pretrained=pretrained
                        )
                        self.assertIs(result, mocked.return_value)

    def test_default_options(self) -> None:
        for name, builder in (
            ("resnet50", "build_resnet50"),
            ("vit_b16", "build_vit_b16"),
        ):
            with self.subTest(name=name), patch.object(models, builder) as mocked:
                models.build_model(name)
                mocked.assert_called_once_with(num_classes=101, pretrained=False)

    def test_unsupported_names(self) -> None:
        for name in ("", "unknown", "vit_b_16"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "Unsupported model name"):
                    models.build_model(name)

    def test_invalid_class_count_propagates(self) -> None:
        for name in ("resnet50", "vit_b16"):
            with self.subTest(name=name), self.assertRaisesRegex(
                ValueError, "num_classes must be positive"
            ):
                models.build_model(name, num_classes=0)

    def test_configured_architectures_and_options(self) -> None:
        config_path = SRC.parent / "configs" / "config.yaml"
        with config_path.open(encoding="utf-8") as file:
            config = yaml.safe_load(file)
        experiments = config["experiments"]
        self.assertEqual(experiments["architectures"], ["resnet50", "vit_b16"])
        self.assertIs(experiments["pretrained"], False)
        for name, builder in (
            ("resnet50", "build_resnet50"),
            ("vit_b16", "build_vit_b16"),
        ):
            with self.subTest(name=name), patch.object(models, builder) as mocked:
                models.build_model(
                    name,
                    num_classes=config["project"]["num_classes"],
                    pretrained=experiments["pretrained"],
                )
                mocked.assert_called_once_with(num_classes=config["project"]["num_classes"], pretrained=experiments["pretrained"])


class FinetuningUtilitiesTests(unittest.TestCase):
    """Verify exact trainable parameter sets and counts for both architectures."""

    def test_freeze_unfreeze_and_counts(self) -> None:
        module = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2, bias=False))
        total = 3 * 4 + 4 + 4 * 2
        self.assertEqual(models.count_parameters(module), (total, total))
        models.freeze_parameters(module)
        self.assertEqual(models.count_parameters(module), (total, 0))
        models.unfreeze_parameters(module[0])
        self.assertEqual(models.count_parameters(module), (total, 16))
        models.unfreeze_parameters(module)
        self.assertEqual(models.count_parameters(module), (total, total))

    def test_all_regimes_and_transitions(self) -> None:
        config_path = SRC.parent / "configs" / "config.yaml"
        with config_path.open(encoding="utf-8") as file:
            config = yaml.safe_load(file)
        regimes = config["experiments"]["regimes"]
        self.assertEqual(
            set(regimes),
            {"from_scratch", "frozen_backbone", "partial_finetuning",
             "full_finetuning"},
        )
        for name, head_prefix, partial_prefix in (
            ("resnet50", "fc.", "layer4."),
            ("vit_b16", "heads.", "encoder.layers.encoder_layer_11."),
        ):
            model = models.build_model(name, pretrained=False)
            parameters = dict(model.named_parameters())
            total = sum(parameter.numel() for parameter in parameters.values())
            for regime in (
                "frozen_backbone", "partial_finetuning", "from_scratch",
                "partial_finetuning", "full_finetuning", "frozen_backbone",
            ):
                with self.subTest(model=name, regime=regime):
                    models.configure_finetuning(model, regime)
                    if regime in {"from_scratch", "full_finetuning"}:
                        expected = set(parameters)
                    elif regime == "frozen_backbone":
                        expected = {
                            key for key in parameters
                            if key.startswith(head_prefix)
                        }
                    else:
                        expected = {
                            key for key in parameters
                            if key.startswith((head_prefix, partial_prefix))
                        }
                    self.assertTrue(expected)
                    actual = {
                        key for key, parameter in parameters.items()
                        if parameter.requires_grad
                    }
                    self.assertEqual(actual, expected)
                    expected_count = sum(
                        parameters[key].numel() for key in expected
                    )
                    self.assertEqual(
                        models.count_parameters(model), (total, expected_count)
                    )
            del model, parameters

    def test_invalid_inputs_do_not_change_parameter_state(self) -> None:
        model = models.build_resnet50(pretrained=False)
        models.configure_finetuning(model, "frozen_backbone")
        before = {
            name: parameter.requires_grad
            for name, parameter in model.named_parameters()
        }
        with self.assertRaisesRegex(ValueError, "Unsupported training regime"):
            models.configure_finetuning(model, "unknown")
        self.assertEqual(
            before,
            {name: p.requires_grad for name, p in model.named_parameters()},
        )
        unsupported = nn.Linear(2, 2)
        with self.assertRaisesRegex(ValueError, "Unsupported model type"):
            models.configure_finetuning(unsupported, "full_finetuning")
        self.assertTrue(all(p.requires_grad for p in unsupported.parameters()))


if __name__ == "__main__":
    unittest.main()
