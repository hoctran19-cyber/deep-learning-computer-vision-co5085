# Data modules

Detection preparation is implemented in `scripts/prepare_detection_data.py`.

Recognition decoding/cropping is deliberately not guessed in this first commit.
Before implementing it, validate the official CCPD province/alphabet/digit mapping
against the dataset's naming convention and freeze the manifest-derived labels.
