"""Public data types for TatVTON pipeline inputs and outputs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Union

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class PointPrompt:
    """SAM point prompt — one or more (x, y) coordinates with labels.

    Attributes:
        coords: Sequence of ``(x, y)`` pixel coordinates.
        labels: Per-point label (1 = foreground, 0 = background).
            Defaults to all-foreground when omitted.
    """

    coords: Sequence[tuple[int, int]]
    labels: Sequence[int] | None = None

    def __post_init__(self) -> None:
        if not self.coords:
            raise ValueError("PointPrompt requires at least one coordinate.")
        if self.labels is not None and len(self.labels) != len(self.coords):
            raise ValueError(
                f"coords length ({len(self.coords)}) != labels length ({len(self.labels)})"
            )


@dataclass(frozen=True)
class BBoxPrompt:
    """SAM bounding-box prompt — axis-aligned rectangle.

    Attributes:
        bbox: ``(x_min, y_min, x_max, y_max)`` in pixel coordinates.
    """

    bbox: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        x_min, y_min, x_max, y_max = self.bbox
        if x_min >= x_max or y_min >= y_max:
            raise ValueError(
                f"Invalid bbox: x_min({x_min}) >= x_max({x_max}) "
                f"or y_min({y_min}) >= y_max({y_max})"
            )


RegionPrompt = Union[PointPrompt, BBoxPrompt]


@dataclass(frozen=True)
class MaskResult:
    """Result from skin mask extraction.

    Attributes:
        mask: Boolean mask array of shape ``(H, W)``.
        score: Confidence score from SAM (0.0-1.0).
    """

    mask: np.ndarray
    score: float


@dataclass(frozen=True)
class PipelineOutput:
    """Final output from :class:`TatVTONPipeline`.

    Attributes:
        image: Composited result at original resolution.
        mask: Binary mask used for inpainting (original resolution).
        raw_inpainted: Raw inpainting output before compositing.
        seed: Random seed used for reproducibility.
    """

    image: Image.Image
    mask: Image.Image
    raw_inpainted: Image.Image
    seed: int
    metadata: dict = field(default_factory=dict)
