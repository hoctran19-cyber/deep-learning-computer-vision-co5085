"""Kiểm tra khả thi GPU: một mini-batch ViT-B/16 full fine-tuning qua train_one_epoch()."""

import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch import nn

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from models import build_vit_b16, configure_finetuning
from optim import build_optimizer
from train import train_one_epoch

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("15.3 NOT RUN: CUDA is not available")
        return 2

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config["training"]["regime"] = "full_finetuning"
    use_amp = True
    device = torch.device("cuda")
    model = build_vit_b16(num_classes=101, pretrained=False)
    configure_finetuning(model, "full_finetuning")
    model.to(device)
    optimizer = build_optimizer(model, config)

    generator = torch.Generator().manual_seed(0)
    images = torch.randn(args.batch_size, 3, 224, 224, generator=generator)
    labels = torch.randint(0, 101, (args.batch_size,), generator=generator)

    torch.cuda.reset_peak_memory_stats()
    # Một batch duy nhất nên train_one_epoch thực hiện đúng một optimizer step.
    metrics = train_one_epoch(
        model, [(images, labels)], nn.CrossEntropyLoss(), optimizer, device,
        use_amp=use_amp,
    )
    torch.cuda.synchronize()
    peak_mb = torch.cuda.max_memory_allocated() / 2**20
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"batch_size={args.batch_size} samples={metrics['num_samples']}")
    print(f"loss={metrics['loss']:.4f} amp={use_amp}")
    print(f"peak_vram_mb={peak_mb:.2f}")
    print("15.3 PASS (one mini-batch only; not proof of full-epoch memory safety)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
