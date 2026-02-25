"""TatVTONPipeline — main orchestrator for tattoo virtual try-on."""

from __future__ import annotations

import logging

import torch
from PIL import Image

from ttvton.config import TatVTONConfig
from ttvton.models.model_loader import ModelLoader
from ttvton.pipeline.inpainting_engine import InpaintingEngine
from ttvton.postprocessing.compositing import Compositor
from ttvton.preprocessing.input_validator import InputValidator
from ttvton.preprocessing.skin_mask_extractor import SkinMaskExtractor
from ttvton.types import PipelineOutput, RegionPrompt
from ttvton.utils.image import resize_for_pipeline
from ttvton.utils.mask import dilate_mask, mask_to_pil

logger = logging.getLogger(__name__)

_CLIP_IMAGE_SIZE = 224


class TatVTONPipeline:
    """End-to-end tattoo virtual try-on pipeline.

    Orchestrates preprocessing (SAM mask extraction), diffusion-based
    inpainting (SDXL + ControlNet + IP-Adapter), and post-processing
    (alpha-blended compositing).

    Usage::

        from ttvton import TatVTONPipeline, TatVTONConfig, PointPrompt

        pipe = TatVTONPipeline()
        result = pipe(
            body_image=body,
            tattoo_image=tattoo,
            region=PointPrompt(coords=[(300, 400)]),
        )
        result.image.save("output.png")
    """

    def __init__(self, config: TatVTONConfig | None = None) -> None:
        self._config = config or TatVTONConfig()
        self._model_loader = ModelLoader(self._config)
        self._inpainting_engine = InpaintingEngine(self._config, self._model_loader)
        self._compositor = Compositor(
            dilation_pixels=self._config.mask_dilation_pixels,
            feather_sigma=self._config.mask_feather_sigma,
        )

    @property
    def config(self) -> TatVTONConfig:
        """Return the current pipeline configuration."""
        return self._config

    @staticmethod
    def from_pretrained(
        repo_id: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ) -> TatVTONPipeline:
        """Create a pipeline from a HF Hub repository.

        Args:
            repo_id: Hub repo containing TatVTON-specific model config.
            device: Target device.
            dtype: Model precision.

        Returns:
            Configured :class:`TatVTONPipeline`.
        """
        config = TatVTONConfig(
            controlnet_model_id=repo_id,
            device=device,
            dtype=dtype,
        )
        return TatVTONPipeline(config)

    def __call__(
        self,
        body_image: Image.Image,
        tattoo_image: Image.Image,
        region: RegionPrompt,
        *,
        strength: float | None = None,
        ip_adapter_scale: float | None = None,
        num_inference_steps: int | None = None,
        seed: int | None = None,
        prompt: str | None = None,
    ) -> PipelineOutput:
        """Run the full tattoo virtual try-on pipeline.

        Args:
            body_image: Body photo (any resolution, RGB).
            tattoo_image: Tattoo design (any resolution, RGB or RGBA).
            region: :class:`PointPrompt` or :class:`BBoxPrompt` for placement.
            strength: Denoising strength (per-call override).
            ip_adapter_scale: IP-Adapter influence (per-call override).
            num_inference_steps: Denoising steps (per-call override).
            seed: Random seed (per-call override).
            prompt: Optional text prompt.

        Returns:
            :class:`PipelineOutput` with composited image and metadata.
        """
        cfg = self._config

        # --- Validate inputs ---
        InputValidator.validate_call(
            body_image, tattoo_image, region, strength, ip_adapter_scale, num_inference_steps
        )

        _seed = seed if seed is not None else cfg.seed
        if _seed is None:
            _seed = torch.randint(0, 2**32, (1,)).item()

        # --- Phase 1: Preprocessing (SAM mask extraction) ---
        logger.info("Phase 1: Extracting skin mask with SAM")
        sam_predictor = self._model_loader.load_sam_predictor()
        extractor = SkinMaskExtractor(sam_predictor)
        mask_result = extractor.extract(body_image, region)
        logger.info("Mask extracted (score=%.3f)", mask_result.score)

        # Unload SAM to free VRAM
        extractor.unload()
        self._model_loader.unload_component("sam")

        # --- Phase 1b: DensePose (optional) ---
        if cfg.use_densepose:
            logger.info("Phase 1b: Extracting DensePose UV map")
            from ttvton.preprocessing.densepose_extractor import DensePoseExtractor

            dp = DensePoseExtractor(
                config_path="",  # TODO: make configurable
                weights_url="",
                device=cfg.device,
            )
            control_image = dp.extract(body_image)
            dp.unload()
        else:
            control_image = body_image.convert("RGB")

        # --- Phase 2: Resize to pipeline resolution ---
        logger.info("Phase 2: Resizing to %d", cfg.resolution)
        original_size = body_image.size
        body_resized = resize_for_pipeline(body_image, cfg.resolution)
        mask_dilated = dilate_mask(mask_result.mask, cfg.mask_dilation_pixels)

        mask_pil = mask_to_pil(mask_dilated)
        mask_resized = resize_for_pipeline(mask_pil, cfg.resolution)
        control_resized = resize_for_pipeline(control_image, cfg.resolution)
        tattoo_resized = resize_for_pipeline(
            tattoo_image.convert("RGB"), _CLIP_IMAGE_SIZE
        )

        # --- Phase 3: Diffusion inpainting ---
        logger.info("Phase 3: Running inpainting")
        raw_inpainted = self._inpainting_engine.generate(
            image=body_resized,
            mask_image=mask_resized,
            control_image=control_resized,
            tattoo_image=tattoo_resized,
            prompt=prompt,
            strength=strength,
            ip_adapter_scale=ip_adapter_scale,
            num_inference_steps=num_inference_steps,
            seed=_seed,
        )

        # --- Phase 4: Compositing ---
        logger.info("Phase 4: Compositing")
        import numpy as np

        mask_at_pipeline_res = np.asarray(mask_resized) > 127
        composited = self._compositor.composite(
            body_resized, raw_inpainted, mask_at_pipeline_res
        )

        # Resize back to original
        final_image = composited.resize(original_size, Image.Resampling.LANCZOS)
        final_mask = mask_pil.resize(original_size, Image.Resampling.NEAREST)

        logger.info("Done (seed=%d)", _seed)

        return PipelineOutput(
            image=final_image,
            mask=final_mask,
            raw_inpainted=raw_inpainted,
            seed=_seed,
        )

    def unload(self) -> None:
        """Release all loaded models and free GPU memory."""
        self._inpainting_engine.unload()
        self._model_loader.unload_all()
        logger.info("Pipeline fully unloaded")
