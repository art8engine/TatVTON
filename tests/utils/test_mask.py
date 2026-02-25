"""Tests for mask manipulation utilities."""

from __future__ import annotations

import numpy as np
from PIL import Image

from tatvton.utils.mask import (
    dilate_mask,
    feather_mask,
    invert_mask,
    mask_to_pil,
    pil_to_mask,
)


class TestDilate:
    def test_expands_mask(self, dummy_mask: np.ndarray) -> None:
        dilated = dilate_mask(dummy_mask, pixels=5)
        assert dilated.sum() > dummy_mask.sum()
        assert dilated.dtype == bool

    def test_zero_pixels_unchanged(self, dummy_mask: np.ndarray) -> None:
        result = dilate_mask(dummy_mask, pixels=0)
        np.testing.assert_array_equal(result, dummy_mask)

    def test_negative_pixels_unchanged(self, dummy_mask: np.ndarray) -> None:
        result = dilate_mask(dummy_mask, pixels=-1)
        np.testing.assert_array_equal(result, dummy_mask)


class TestFeather:
    def test_output_range(self, dummy_mask: np.ndarray) -> None:
        feathered = feather_mask(dummy_mask, sigma=3.0)
        assert feathered.dtype == np.float32
        assert feathered.min() >= 0.0
        assert feathered.max() <= 1.0

    def test_zero_sigma(self, dummy_mask: np.ndarray) -> None:
        result = feather_mask(dummy_mask, sigma=0.0)
        assert result.dtype == np.float32


class TestInvert:
    def test_invert(self, dummy_mask: np.ndarray) -> None:
        inverted = invert_mask(dummy_mask)
        np.testing.assert_array_equal(inverted, ~dummy_mask)


class TestPilConversion:
    def test_bool_mask_to_pil(self, dummy_mask: np.ndarray) -> None:
        pil = mask_to_pil(dummy_mask)
        assert pil.mode == "L"
        assert pil.size == (256, 256)

    def test_roundtrip(self, dummy_mask: np.ndarray) -> None:
        pil = mask_to_pil(dummy_mask)
        recovered = pil_to_mask(pil)
        np.testing.assert_array_equal(recovered, dummy_mask)
