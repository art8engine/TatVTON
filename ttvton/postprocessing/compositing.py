"""Alpha-blending compositor with feathered mask support."""

from __future__ import annotations

import numpy as np
from PIL import Image

from ttvton.utils.image import numpy_to_pil, pil_to_numpy
from ttvton.utils.mask import dilate_mask, feather_mask


class Compositor:
    """Blend the inpainted result back onto the original body image.

    Workflow:
      1. Dilate the binary mask to cover edge artefacts.
      2. Feather the mask for smooth transitions.
      3. Alpha-blend: ``output = original * (1 - a) + inpainted * a``.
      4. Resize back to original resolution.
    """

    def __init__(
        self,
        dilation_pixels: int = 10,
        feather_sigma: float = 5.0,
    ) -> None:
        self._dilation_pixels = dilation_pixels
        self._feather_sigma = feather_sigma

    def composite(
        self,
        original: Image.Image,
        inpainted: Image.Image,
        mask: np.ndarray,
    ) -> Image.Image:
        """Blend *inpainted* onto *original* using *mask*.

        Args:
            original: Original body photo at pipeline resolution.
            inpainted: Raw output from the inpainting engine at pipeline resolution.
            mask: Boolean mask at pipeline resolution ``(H, W)``.

        Returns:
            Composited PIL image at the same resolution as inputs.
        """
        dilated = dilate_mask(mask, self._dilation_pixels)
        alpha = feather_mask(dilated, self._feather_sigma)

        orig_arr = pil_to_numpy(original).astype(np.float32)
        inpaint_arr = pil_to_numpy(inpainted).astype(np.float32)

        alpha_3d = alpha[:, :, np.newaxis]
        blended = orig_arr * (1.0 - alpha_3d) + inpaint_arr * alpha_3d

        return numpy_to_pil(blended)

    def composite_with_resize(
        self,
        original_full: Image.Image,
        inpainted: Image.Image,
        mask: np.ndarray,
        original_resized: Image.Image,
    ) -> Image.Image:
        """Composite at pipeline resolution, then resize to original dimensions.

        Args:
            original_full: Original body photo at full resolution.
            inpainted: Inpainting output at pipeline resolution.
            mask: Boolean mask at pipeline resolution.
            original_resized: Original photo resized to pipeline resolution.

        Returns:
            Composited image at *original_full* resolution.
        """
        composited = self.composite(original_resized, inpainted, mask)
        return composited.resize(original_full.size, Image.Resampling.LANCZOS)
