"""Tests for image conversion utilities."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from ttvton.utils.image import (
    extract_alpha_mask,
    numpy_to_pil,
    pil_to_numpy,
    pil_to_tensor,
    resize_for_pipeline,
    tensor_to_pil,
)


class TestPilNumpyRoundtrip:
    def test_rgb_roundtrip(self, dummy_body_image: Image.Image) -> None:
        arr = pil_to_numpy(dummy_body_image)
        assert arr.shape == (256, 256, 3)
        assert arr.dtype == np.uint8
        result = numpy_to_pil(arr)
        assert result.mode == "RGB"
        assert result.size == (256, 256)

    def test_rgba_converted_to_rgb(self, dummy_tattoo_image: Image.Image) -> None:
        arr = pil_to_numpy(dummy_tattoo_image)
        assert arr.shape[2] == 3  # RGBA → RGB


class TestPilTensorRoundtrip:
    def test_shape_and_range(self, dummy_body_image: Image.Image) -> None:
        t = pil_to_tensor(dummy_body_image)
        assert t.shape == (1, 3, 256, 256)
        assert t.min() >= 0.0
        assert t.max() <= 1.0

    def test_roundtrip(self, dummy_body_image: Image.Image) -> None:
        t = pil_to_tensor(dummy_body_image)
        img = tensor_to_pil(t)
        assert img.size == (256, 256)


class TestResize:
    def test_square(self, dummy_body_image: Image.Image) -> None:
        resized = resize_for_pipeline(dummy_body_image, 128)
        assert resized.size == (128, 128)

    def test_tuple(self, dummy_body_image: Image.Image) -> None:
        resized = resize_for_pipeline(dummy_body_image, (64, 128))
        assert resized.size == (64, 128)


class TestAlpha:
    def test_rgba_has_alpha(self, dummy_tattoo_image: Image.Image) -> None:
        alpha = extract_alpha_mask(dummy_tattoo_image)
        assert alpha is not None
        assert alpha.mode == "L"

    def test_rgb_no_alpha(self, dummy_body_image: Image.Image) -> None:
        assert extract_alpha_mask(dummy_body_image) is None
