# A1.2 — CNN vs Transformer Experiments

This directory contains the implementation and experiment workflow for Assignment A1.2.

## Execution architecture

- **Local Mac (Intel):** edit source code, validate small components, manage Git.
- **Google Colab:** run Food-101 preparation, model training, validation, and final test evaluation.
- **Google Drive:** store the dataset, checkpoints, logs, metrics, and generated figures.
- **GitHub:** version source code, configuration, notebooks, and reports. Do not commit the Food-101 image archive or large checkpoints.

## Planned experiments

Two architectures are evaluated under four training regimes each:

| Architecture | Training regimes |
|---|---|
| ResNet-50 | from scratch, frozen backbone, partial fine-tuning (layer4 + head), full fine-tuning |
| ViT-B/16 | from scratch, frozen backbone, partial fine-tuning (last encoder block + head), full fine-tuning |

The experiment protocol, split strategy, preprocessing, and metrics are documented in [A1-1](../A1-1/README.md). The implementation must follow that protocol unless a documented change is approved.

## Repository layout

- `configs/config.yaml`: shared experiment configuration.
- `notebooks/A1_2_experiment.ipynb`: Google Colab entry point.
- `src/`: reusable dataset, transform, model, training, evaluation, and utility modules.
- `scripts/`: command-line helpers.
- `outputs/`: local output structure; large generated files are ignored by Git.

## Colab workflow

1. Open `notebooks/A1_2_experiment.ipynb` in Google Colab.
2. Select a GPU runtime when available and verify it with `torch.cuda.is_available()`.
3. Mount Google Drive and set the project/data/output paths in the notebook.
4. Clone or pull the `assignment_A1` branch.
5. Implement and validate the pipeline before running the full experiments.
6. Save checkpoints and experiment artifacts to Google Drive.
7. Export compact result tables and figures for the report.

## Dataset and licensing

Food-101 images are not included in this repository. Review the dataset usage and licensing notes in [A1-1/README.md](../A1-1/README.md) before downloading or redistributing data. Do not upload the original images to GitHub.

## Validate the Food-101 manifest

From the repository root, validate the manifest independently against the
official metadata (no image files or third-party packages are needed):

```bash
python3 A1-2/src/validate_manifest.py \
  --dataset-root A1-2/data/food-101-metadata \
  --manifest A1-2/data/manifests/food101_split_seed42.csv
```

`--dataset-root` must contain `meta/classes.txt`, `meta/train.json`, and
`meta/test.json`. The validator checks class order and labels, paths, split
counts, duplicates, and exact official membership. It prints each check and
returns a nonzero exit code if validation fails. It does not verify that image
files exist.

Run the synthetic integration tests with:

```bash
python3 -m unittest discover -s A1-2/tests -p 'test_validate_manifest.py'
```

## Current status

This commit establishes the A1.2 project scaffold only. Dataset loading, split-manifest generation, training, evaluation, and experiment results have **not** been implemented yet.
