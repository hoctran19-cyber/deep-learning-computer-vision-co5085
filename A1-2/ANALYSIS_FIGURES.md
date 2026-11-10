# A1-2 analysis figures

Run the generator from the repository root:

```bash
python3 A1-2/scripts/generate_analysis_figures.py
```

This creates training-history and experiment-comparison figures from the
existing `training_log.jsonl` and `summary.json` files. To also create the
sample grid, confusion matrix, and prediction examples, provide the full
Food-101 root and the existing selected-model best checkpoint:

```bash
python3 A1-2/scripts/generate_analysis_figures.py \
  --dataset-root /path/to/food-101 \
  --checkpoint /path/to/resnet50_full_finetune/best_checkpoint.pt
```

The dataset root must contain both `meta/classes.txt` and `images/`. The
checkpoint is checked against the SHA-256 and best epoch recorded for
`resnet50_full_finetune`. Inference on the test split is used only to build the
confusion matrix and select example images; aggregate test metrics are neither
recomputed nor saved. The existing final test metrics are left unchanged.

Generated files go to `A1-2/outputs/figures/`:

| Figure | What it shows | Interpretation |
|---|---|---|
| `training_loss.png` | Training and validation loss for each experiment by epoch. | Compare optimization progress and look for a widening train/validation gap, which can indicate overfitting. |
| `training_accuracy.png` | Training and validation accuracy for each experiment by epoch. | Shows fit progress alongside generalization; compare trends rather than only the final epoch because runs have different lengths. |
| `validation_macro_f1_curves.png` | Validation Macro-F1 by epoch. | Tracks class-balanced validation performance and the checkpoint-selection objective across runs. |
| `validation_accuracy_comparison.png` | Best validation accuracy recorded in each experiment summary. | Compares the selected best-epoch accuracy across the seven experiments. |
| `validation_macro_f1_comparison.png` | Best validation Macro-F1 recorded in each experiment summary. | Compares the class-balanced selection metric across the seven experiments. |
| `training_time_comparison.png` | Recorded experiment wall time in minutes. | Provides a runtime comparison; these values include validation and artifact I/O, not training-only time. |
| `food101_sample_grid.png` | Sixteen training images selected deterministically from the manifest, with class labels. | Sanity-checks visual variety and label readability; it is not a model evaluation. |
| `resnet50_full_finetune_confusion_matrix.png` | Raw class-by-class counts on the test split from the selected checkpoint. | The diagonal is correct classification; off-diagonal cells show class confusions. |
| `resnet50_full_finetune_examples.png` | Up to eight correct and eight incorrect test predictions. | Each tile shows Ground Truth and Predicted Label to make representative successes and errors inspectable. |

The history and comparison charts use the metrics present in the experiment
JSON files; no metric values are embedded in the generator. The last three
figures require the original image files, and the final two also require the
selected checkpoint. In the inspected checkout the manifest and class metadata
are present, but the Food-101 `images/` directory and selected checkpoint are
not, so those image-dependent figures remain unavailable until the files are
provided.

In the available comparisons, ResNet50 full fine-tuning leads the experiments
on both validation metrics, while ViT-B/16 from scratch is lowest. The curves
also show a substantial training/validation gap for the fine-tuned ResNet50;
interpret its strong validation result alongside that gap rather than treating
training accuracy alone as evidence of generalization.
