"""Kiểm thử preprocessing theo protocol A1-1."""

import sys
import unittest
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from transforms import build_eval_transform, build_train_transform


class TransformTests(unittest.TestCase):
    """Kiểm tra thứ tự pipeline, shape, normalization và tham số."""

    def test_pipeline_and_normalization(self) -> None:
        image = Image.new("RGB", (320, 260), (255, 0, 128))
        for builder, crop in (
            (build_train_transform, transforms.RandomCrop),
            (build_eval_transform, transforms.CenterCrop),
        ):
            transform = builder()
            self.assertIsInstance(transform.transforms[0], transforms.Resize)
            self.assertEqual(transform.transforms[0].size, 256)
            self.assertIsInstance(transform.transforms[1], crop)
            output = transform(image)
            self.assertEqual(tuple(output.shape), (3, 224, 224))
            self.assertEqual(output.dtype, torch.float32)
            self.assertTrue(torch.allclose(
                output[:, 0, 0],
                torch.tensor([(1 - 0.485) / 0.229, -0.456 / 0.224,
                              (128 / 255 - 0.406) / 0.225]),
                atol=1e-5,
            ))
        self.assertEqual(build_train_transform().transforms[2].p, 0.5)

    def test_configurable_and_eval_deterministic(self) -> None:
        image = Image.new("RGB", (40, 35), (64, 128, 192))
        transform = build_eval_transform(32, 24, mean=(0, 0, 0), std=(1, 1, 1))
        self.assertTrue(torch.equal(transform(image), transform(image)))
        self.assertEqual(tuple(transform(image).shape), (3, 24, 24))
        self.assertAlmostEqual(transform(image)[0, 0, 0].item(), 64 / 255)
        train = build_train_transform(32, 24, 0.0, (0, 0, 0), (1, 1, 1))
        self.assertEqual(train.transforms[2].p, 0.0)
        self.assertEqual(tuple(train(image).shape), (3, 24, 24))

    def test_invalid_parameters(self) -> None:
        for kwargs in (
            {"resize_short_edge": 0}, {"crop_size": 0},
            {"resize_short_edge": 32, "crop_size": 33},
            {"mean": (0, 0)}, {"std": (1, 0, 1)},
            {"mean": (float("nan"), 0, 0)},
            {"horizontal_flip_probability": 1.1},
            {"horizontal_flip_probability": float("nan")},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                build_train_transform(**kwargs)


if __name__ == "__main__":
    unittest.main()
