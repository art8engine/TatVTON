"""Shared test fixtures — dummy images, mock models."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image


@pytest.fixture()
def dummy_body_image() -> Image.Image:
    """256x256 solid-colour body image."""
    return Image.fromarray(
        np.full((256, 256, 3), 180, dtype=np.uint8), mode="RGB"
    )


@pytest.fixture()
def dummy_tattoo_image() -> Image.Image:
    """128x128 tattoo design with alpha channel."""
    arr = np.zeros((128, 128, 4), dtype=np.uint8)
    arr[:, :, :3] = 50  # dark tattoo
    arr[:, :, 3] = 200  # mostly opaque
    return Image.fromarray(arr, mode="RGBA")


@pytest.fixture()
def dummy_mask() -> np.ndarray:
    """256x256 boolean mask with a centred circle."""
    h, w = 256, 256
    y, x = np.ogrid[:h, :w]
    centre = (h // 2, w // 2)
    radius = 50
    return ((x - centre[1]) ** 2 + (y - centre[0]) ** 2) <= radius**2


@pytest.fixture()
def small_body_image() -> Image.Image:
    """64x64 — minimum valid size."""
    return Image.fromarray(
        np.full((64, 64, 3), 200, dtype=np.uint8), mode="RGB"
    )
