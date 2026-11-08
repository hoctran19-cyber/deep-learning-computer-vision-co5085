"""Tensor preprocessing for row-wise image sequences.

This module does not load or download datasets.
"""

from torch import Tensor


def image_to_row_sequence(images: Tensor) -> Tensor:
    """Convert grayscale images from (batch, 1, height, width) to row sequences.

    The resulting tensor has shape (batch, height, width), with rows kept in
    their original top-to-bottom order.
    """
    if images.ndim != 4 or images.shape[1] != 1:
        raise ValueError("Expected grayscale images with shape (batch, 1, height, width).")
    if images.shape[2] == 0 or images.shape[3] == 0:
        raise ValueError("Image height and width must be positive.")
    return images.squeeze(1)
