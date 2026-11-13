# A2-2 — License Plate Detection and Recognition

This folder contains the implementation for the license-plate detection and recognition assignment.

## Experimental protocol

- Dataset: `CCPD_reduced` (25,000 images).
- Split manifest is authoritative; do not create a new random split during training.
- Train/validation: CCPD-Base (18,000 / 2,000).
- Test: held-out challenge subsets (5,000 total). Report aggregate metrics and per-subset results.
- Detector: YOLO26n.
- E1 recognizer: CRNN-BiLSTM-CTC.
- E2 recognizer: PDLPR-inspired recognizer, implemented and documented separately.
- E1/E2 must use the same detector checkpoint, image crops, split and evaluation protocol.

## Data location

Keep images, annotations, crops, checkpoints and training outputs on Google Drive. Do not commit dataset images or model weights.

Expected Drive paths (adjust only if your Drive layout differs):

```text
/content/drive/MyDrive/ComputerVision/A2/CCPD_reduced
/content/drive/MyDrive/ComputerVision/A2/manifest.csv
```

The scripts accept explicit paths and never assume that the dataset is inside this Git repository.

## Quick start (Colab)

```bash
%cd /content
!git clone https://github.com/hoctran19-cyber/deep-learning-computer-vision-co5085.git
%cd deep-learning-computer-vision-co5085/A2-2
!pip install -r requirements.txt
```

Mount Drive, then prepare YOLO detection data:

```bash
!python scripts/prepare_detection_data.py \
  --root /content/drive/MyDrive/ComputerVision/A2/CCPD_reduced \
  --manifest /content/drive/MyDrive/ComputerVision/A2/manifest.csv \
  --output /content/drive/MyDrive/ComputerVision/A2/yolo_detection \
  --link-images
```

Then train YOLO26n using the generated `data.yaml` and the official Ultralytics API/CLI. Record the exact package version, model checkpoint, image size, seed, epochs and best validation checkpoint.

## Important data rules

- The manifest determines each image's split; do not infer or reshuffle splits.
- Bounding boxes are parsed from CCPD filenames and converted to normalized YOLO format.
- The script validates image readability, bounding-box bounds and split counts before declaring success.
- License-plate text decoding is intentionally separate: use the official CCPD character mapping, do not decode by guessing.
- Test annotations must not be used for model selection.

## Folder map

- `scripts/prepare_detection_data.py`: convert manifest-backed CCPD boxes to YOLO labels.
- `configs/detection.yaml`: detector experiment defaults.
- `src/data/`: shared detection/recognition data code (to be extended in subsequent steps).
- `notebooks/`: Colab experiment notebooks.
- `outputs/`: local/Drive run outputs; large artifacts are ignored by Git.
