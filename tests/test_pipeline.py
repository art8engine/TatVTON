"""Tests for TatVTONPipeline input validation and config."""

from __future__ import annotations

import pytest
from PIL import Image

from tatvton import BBoxPrompt, PointPrompt, TatVTONConfig, TatVTONPipeline


class TestTatVTONConfig:
    def test_frozen(self) -> None:
        cfg = TatVTONConfig()
        with pytest.raises(AttributeError):
            cfg.resolution = 512  # type: ignore[misc]

    def test_replace(self) -> None:
        cfg = TatVTONConfig()
        new = cfg.replace(resolution=768, num_inference_steps=20)
        assert new.resolution == 768
        assert new.num_inference_steps == 20
        assert cfg.resolution == 1024  # original unchanged

    def test_defaults(self) -> None:
        cfg = TatVTONConfig()
        assert cfg.resolution == 1024
        assert cfg.strength == 0.85
        assert cfg.use_densepose is False


class TestPipelineInit:
    def test_default_config(self) -> None:
        pipe = TatVTONPipeline()
        assert pipe.config.resolution == 1024

    def test_custom_config(self) -> None:
        cfg = TatVTONConfig(resolution=768)
        pipe = TatVTONPipeline(cfg)
        assert pipe.config.resolution == 768


class TestPointPrompt:
    def test_valid(self) -> None:
        p = PointPrompt(coords=[(100, 200)])
        assert len(p.coords) == 1

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            PointPrompt(coords=[])

    def test_label_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="labels length"):
            PointPrompt(coords=[(1, 2), (3, 4)], labels=[1])


class TestBBoxPrompt:
    def test_valid(self) -> None:
        b = BBoxPrompt(bbox=(10, 20, 100, 200))
        assert b.bbox == (10, 20, 100, 200)

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid bbox"):
            BBoxPrompt(bbox=(100, 200, 10, 20))
