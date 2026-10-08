# CO5085 E1 — Softmax, MLP, and CNN

This project compares three manually trained PyTorch image classifiers on a
shared Fashion-MNIST train/validation/test split: a linear Softmax classifier,
a flattened fully-connected MLP, and a convolutional neural network. MNIST
and CIFAR-10 can also be selected in the configuration.

No data, models, metrics, or figures are generated during setup. Results are
created only when an experiment is deliberately run.

## Project structure

`src/datasets.py` handles deterministic splits and DataLoaders; `src/models.py`
contains the three models; `src/train.py` contains the explicit
`zero_grad → forward → loss → backward → step` loop and CLI; `src/evaluate.py`
contains test/prediction/error helpers; and `src/utils.py` provides seeds,
automatic device selection, checkpoints, and plots. Configuration is in
`configs/config.yaml`, and the notebook is an experiment runner that imports
these modules rather than duplicating them.

## Installation and local Ubuntu development

From the E1 directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Local execution automatically uses CPU on systems without CUDA. Edit source
and configuration with OpenCode, commit locally, then push to GitHub. Data is
stored under the project-relative `data/` directory and is not downloaded by
the setup or smoke checks. Set `dataset.download: true` only when ready to
run a real experiment.

## Run experiments

All commands are run from `E1/`:

```bash
python3 -m src.train --model softmax
python3 -m src.train --model mlp
python3 -m src.train --model cnn
```

Useful overrides are available without editing YAML:

```bash
python3 -m src.train --model cnn --epochs 10 --batch-size 128 --lr 0.001 \
  --config configs/config.yaml
```

The same seed, split, and configuration should be retained across all three
runs for a fair comparison. The default device is selected with
`cuda if torch.cuda.is_available() else cpu`; no CUDA-specific package or GPU
model is required.

## Google Colab workflow

1. Open Colab and select a GPU runtime if desired.
2. Clone the repository (replace the placeholder URL):

```bash
!git clone <REPOSITORY_URL>
%cd <REPOSITORY_DIRECTORY>/E1
!pip install -r requirements.txt
```

3. Set `dataset.download: true` when ready, then run the three commands above
   in Colab. The code automatically uses the available GPU, otherwise CPU.
4. Inspect or download `results/`, then commit and push desired artifacts.

GitHub synchronizes the source/configuration between local Ubuntu and Colab;
Colab is the intended training environment, while the final source and
selected real results can be submitted through GitHub. Do not commit large
checkpoints unless explicitly required.

## Outputs and reproducibility

Each completed run writes real values to `results/metrics/<model>.json`,
updates `results/metrics/model_comparison.csv`, saves plots under
`results/figures/`, and writes its checkpoint under `results/checkpoints/`.
The comparison contains model name, test accuracy, parameter count, best
validation accuracy, and training time. `data/` and checkpoints are ignored
by Git by default; small JSON/CSV/PNG results remain committable.

The configured seed is used for Python, NumPy, PyTorch, the split generator,
and the DataLoader setup. Exact reproducibility can still vary across devices.
