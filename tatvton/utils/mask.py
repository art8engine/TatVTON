"""Mask manipulation utilities — dilation, feathering, inversion."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def dilate_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    """Dilate a boolean mask by *pixels* using a square structuring element.

    Args:
        mask: ``(H, W)`` boolean array.
        pixels: Dilation radius in pixels. Zero or negative returns the mask unchanged.

    Returns:
        New dilated boolean mask (original is not mutated).
    """
    if pixels <= 0:
        return mask.copy()
    pil_mask = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    pil_mask = pil_mask.filter(ImageFilter.MaxFilter(size=2 * pixels + 1))
    return np.asarray(pil_mask) > 127


def feather_mask(mask: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian blur to create a soft-edged (feathered) mask.

    Args:
        mask: ``(H, W)`` boolean or uint8 array.
        sigma: Gaussian blur standard deviation.

    Returns:
        ``(H, W)`` float32 array in ``[0.0, 1.0]``.
    """
    if sigma <= 0:
        return mask.astype(np.float32)
    pil_mask = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    blurred = pil_mask.filter(ImageFilter.GaussianBlur(radius=sigma))
    return np.asarray(blurred, dtype=np.float32) / 255.0


def invert_mask(mask: np.ndarray) -> np.ndarray:
    """Invert a boolean mask."""
    return ~mask


def mask_to_pil(mask: np.ndarray) -> Image.Image:
    """Convert a boolean or float mask to a PIL ``'L'`` mode image."""
    if mask.dtype == bool:
        arr = (mask * 255).astype(np.uint8)
    elif mask.dtype == np.float32 or mask.dtype == np.float64:
        arr = (np.clip(mask, 0.0, 1.0) * 255).astype(np.uint8)
    else:
        arr = mask.astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def pil_to_mask(image: Image.Image, threshold: int = 127) -> np.ndarray:
    """Convert a PIL ``'L'`` image to a boolean mask."""
    return np.asarray(image.convert("L")) > threshold
