"""Configuration and output paths for the ResNet-50 fine-tuning runs."""

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any


RESNET50_REGIMES = {
    "resnet50_frozen": "frozen_backbone",
    "resnet50_partial": "partial_finetuning",
    "resnet50_full_finetune": "full_finetuning",
}


def resnet50_experiment_configs(
    base_config: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Create independent effective configs, checking the required run matrix."""
    runs = base_config["experiments"]["resnet50_runs"]
    if not isinstance(runs, Mapping) or set(runs) != set(RESNET50_REGIMES):
        raise ValueError("resnet50_runs must contain the three ResNet-50 experiments")
    configs = {}
    for name, regime in RESNET50_REGIMES.items():
        run = runs[name]
        if (
            not isinstance(run, Mapping)
            or run.get("architecture") != "resnet50"
            or run.get("pretrained") is not True
            or run.get("regime") != regime
        ):
            raise ValueError(f"{name}: invalid ResNet-50 experiment configuration")
        overrides = run.get("training", {})
        if not isinstance(overrides, Mapping) or "regime" in overrides:
            raise ValueError(f"{name}: invalid training overrides")
        config = copy.deepcopy(dict(base_config))
        config["training"].update(copy.deepcopy(dict(overrides)))
        config["training"]["regime"] = regime
        config["experiments"]["pretrained"] = True
        config["experiment"] = {
            "name": name,
            "architecture": "resnet50",
            "pretrained": True,
            "regime": regime,
        }
        configs[name] = config
    return configs


def resnet50_output_paths(output_base: Path) -> dict[str, dict[str, Path]]:
    """Resolve isolated artifact paths and reject any existing run directory."""
    output_base = Path(output_base)
    paths = {}
    for name in RESNET50_REGIMES:
        directory = output_base / name
        if directory.exists() or directory.is_symlink():
            raise FileExistsError(f"Experiment output already exists: {directory}")
        paths[name] = {
            "directory": directory,
            "best_checkpoint": directory / "best_checkpoint.pt",
            "last_checkpoint": directory / "last_checkpoint.pt",
            "training_log": directory / "training_log.jsonl",
            "effective_config": directory / "effective_config.yaml",
            "summary": directory / "summary.json",
        }
    return paths
