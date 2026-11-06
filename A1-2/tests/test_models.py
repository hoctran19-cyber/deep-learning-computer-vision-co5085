"""Unit tests for Food-101 model builders."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torchvision.models import ResNet50_Weights
from torchvision.models.resnet import ResNet


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


if __name__ == "__main__":
    unittest.main()
