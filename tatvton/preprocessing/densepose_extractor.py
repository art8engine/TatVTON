"""DensePose UV-map extraction (optional dependency)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from tatvton.utils.image import pil_to_numpy
from tatvton.utils.imports import require_densepose

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class DensePoseExtractor:
    """Extract DensePose IUV maps for ControlNet conditioning.

    Requires ``pip install tatvton[densepose]``.
    """

    def __init__(self, config_path: str, weights_url: str, device: str = "cuda") -> None:
        require_densepose()
        self._device = device
        self._predictor = self._build_predictor(config_path, weights_url, device)

    def extract(self, image: Image.Image) -> Image.Image:
        """Run DensePose inference and return an IUV visualisation as PIL RGB.

        Args:
            image: Body photo.

        Returns:
            ``(H, W, 3)`` PIL image encoding the IUV map.
        """
        img_array = pil_to_numpy(image)
        outputs = self._predictor(img_array)

        iuv_map = self._outputs_to_iuv(outputs, img_array.shape[:2])
        return Image.fromarray(iuv_map, mode="RGB")

    def unload(self) -> None:
        """Release model resources."""
        del self._predictor

    @staticmethod
    def _build_predictor(config_path: str, weights_url: str, device: str) -> Any:
        """Initialise a detectron2-based DensePose predictor."""
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor

        cfg = get_cfg()
        cfg.merge_from_file(config_path)
        cfg.MODEL.WEIGHTS = weights_url
        cfg.MODEL.DEVICE = device
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        return DefaultPredictor(cfg)

    @staticmethod
    def _outputs_to_iuv(
        outputs: Any, shape: tuple[int, int]
    ) -> np.ndarray:
        """Convert detectron2 DensePose outputs to a ``(H, W, 3)`` uint8 IUV array."""
        h, w = shape
        iuv = np.zeros((h, w, 3), dtype=np.uint8)

        if not outputs["instances"].has("pred_densepose"):
            logger.warning("No DensePose predictions found; returning empty IUV map.")
            return iuv

        dp = outputs["instances"].pred_densepose
        boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()

        for i in range(len(dp)):
            x1, y1, x2, y2 = boxes[i].astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            result_i = dp[i].labels.cpu().numpy().astype(np.uint8)
            result_u = (dp[i].uv[0].cpu().numpy() * 255).astype(np.uint8)
            result_v = (dp[i].uv[1].cpu().numpy() * 255).astype(np.uint8)

            bh, bw = y2 - y1, x2 - x1
            if bh <= 0 or bw <= 0:
                continue

            from PIL import Image as _Img

            result_i_resized = np.array(
                _Img.fromarray(result_i).resize((bw, bh), _Img.Resampling.NEAREST)
            )
            result_u_resized = np.array(
                _Img.fromarray(result_u).resize((bw, bh), _Img.Resampling.BILINEAR)
            )
            result_v_resized = np.array(
                _Img.fromarray(result_v).resize((bw, bh), _Img.Resampling.BILINEAR)
            )

            iuv[y1:y2, x1:x2, 0] = result_i_resized
            iuv[y1:y2, x1:x2, 1] = result_u_resized
            iuv[y1:y2, x1:x2, 2] = result_v_resized

        return iuv
