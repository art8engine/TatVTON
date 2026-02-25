"""SAM 2 based skin mask extraction."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from tatvton.types import BBoxPrompt, MaskResult, PointPrompt, RegionPrompt
from tatvton.utils.image import pil_to_numpy

if TYPE_CHECKING:
    from sam2.sam2_image_predictor import SAM2ImagePredictor

logger = logging.getLogger(__name__)


class SkinMaskExtractor:
    """Extract skin region masks using SAM 2.

    The extractor wraps a :class:`SAM2ImagePredictor` and converts
    :class:`RegionPrompt` objects into SAM-compatible inputs.
    """

    def __init__(self, predictor: SAM2ImagePredictor) -> None:
        self._predictor = predictor

    def extract(self, image: Image.Image, region: RegionPrompt) -> MaskResult:
        """Run SAM prediction and return the best mask.

        Args:
            image: Body photo as a PIL image.
            region: Point or bounding-box prompt for the tattoo area.

        Returns:
            :class:`MaskResult` with the highest-confidence mask.
        """
        img_array = pil_to_numpy(image)
        self._predictor.set_image(img_array)

        point_coords, point_labels, box = self._prepare_prompt(region)

        masks, scores, _ = self._predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=True,
        )

        best_idx = int(np.argmax(scores))
        logger.debug("SAM mask scores: %s (selected idx=%d)", scores, best_idx)

        return MaskResult(
            mask=masks[best_idx].astype(bool),
            score=float(scores[best_idx]),
        )

    def refine(
        self,
        image: Image.Image,
        region: RegionPrompt,
        previous: MaskResult,
    ) -> MaskResult:
        """Re-run SAM with a previous mask as additional guidance.

        Args:
            image: Body photo.
            region: Original region prompt.
            previous: Previous extraction result for mask-guided refinement.

        Returns:
            Refined :class:`MaskResult`.
        """
        img_array = pil_to_numpy(image)
        self._predictor.set_image(img_array)

        point_coords, point_labels, box = self._prepare_prompt(region)

        masks, scores, _ = self._predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            mask_input=previous.mask[None, :, :].astype(np.float32),
            multimask_output=False,
        )

        return MaskResult(
            mask=masks[0].astype(bool),
            score=float(scores[0]),
        )

    def unload(self) -> None:
        """Release the underlying SAM model from memory."""
        del self._predictor

    @staticmethod
    def _prepare_prompt(
        region: RegionPrompt,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """Convert a :class:`RegionPrompt` to SAM predictor arguments."""
        if isinstance(region, PointPrompt):
            coords = np.array(region.coords, dtype=np.float32)
            labels = (
                np.array(region.labels, dtype=np.int32)
                if region.labels is not None
                else np.ones(len(region.coords), dtype=np.int32)
            )
            return coords, labels, None
        if isinstance(region, BBoxPrompt):
            box = np.array(region.bbox, dtype=np.float32)
            return None, None, box
        raise TypeError(f"Unsupported region type: {type(region).__name__}")
