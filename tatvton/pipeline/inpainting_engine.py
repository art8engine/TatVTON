"""SDXL + ControlNet + IP-Adapter inpainting engine."""

from __future__ import annotations

import logging
from typing import Any

import torch
from PIL import Image

from tatvton.config import TatVTONConfig
from tatvton.models.model_loader import ModelLoader

logger = logging.getLogger(__name__)


class InpaintingEngine:
    """Wraps the SDXL ControlNet inpainting pipeline with IP-Adapter.

    Responsible for running the diffusion inference step. All image inputs
    must already be resized to the pipeline resolution before calling
    :meth:`generate`.
    """

    def __init__(self, config: TatVTONConfig, model_loader: ModelLoader) -> None:
        self._config = config
        self._model_loader = model_loader
        self._pipe: Any | None = None

    def _ensure_loaded(self) -> Any:
        """Lazy-load the diffusion pipeline on first call."""
        if self._pipe is None:
            self._pipe = self._model_loader.load_inpainting_pipeline()
        return self._pipe

    def generate(
        self,
        image: Image.Image,
        mask_image: Image.Image,
        control_image: Image.Image,
        tattoo_image: Image.Image,
        *,
        prompt: str | None = None,
        strength: float | None = None,
        guidance_scale: float | None = None,
        controlnet_conditioning_scale: float | None = None,
        ip_adapter_scale: float | None = None,
        num_inference_steps: int | None = None,
        seed: int | None = None,
    ) -> Image.Image:
        """Run SDXL inpainting with ControlNet and IP-Adapter conditioning.

        Args:
            image: Body photo at pipeline resolution (e.g. 1024x1024).
            mask_image: Binary mask at pipeline resolution (``'L'`` mode).
            control_image: ControlNet conditioning image at pipeline resolution.
            tattoo_image: Tattoo design resized for CLIP (e.g. 224x224).
            prompt: Optional text prompt (default: generic tattoo prompt).
            strength: Denoising strength override.
            guidance_scale: Classifier-free guidance scale override.
            controlnet_conditioning_scale: ControlNet scale override.
            ip_adapter_scale: IP-Adapter scale override.
            num_inference_steps: Number of denoising steps override.
            seed: Random seed for reproducibility.

        Returns:
            Inpainted PIL image at pipeline resolution.
        """
        pipe = self._ensure_loaded()
        cfg = self._config

        _strength = strength if strength is not None else cfg.strength
        _guidance = guidance_scale if guidance_scale is not None else cfg.guidance_scale
        _cn_scale = (
            controlnet_conditioning_scale
            if controlnet_conditioning_scale is not None
            else cfg.controlnet_conditioning_scale
        )
        _steps = (
            num_inference_steps
            if num_inference_steps is not None
            else cfg.num_inference_steps
        )
        _prompt = prompt or "a photorealistic tattoo on skin, high quality, detailed"

        if ip_adapter_scale is not None:
            pipe.set_ip_adapter_scale(ip_adapter_scale)

        _seed = seed if seed is not None else cfg.seed
        generator = (
            torch.Generator(device=self._model_loader.device).manual_seed(_seed)
            if _seed is not None
            else None
        )

        logger.info(
            "Generating: steps=%d, strength=%.2f, guidance=%.1f, cn_scale=%.2f",
            _steps,
            _strength,
            _guidance,
            _cn_scale,
        )

        result = pipe(
            prompt=_prompt,
            image=image,
            mask_image=mask_image,
            control_image=control_image,
            ip_adapter_image=tattoo_image,
            strength=_strength,
            guidance_scale=_guidance,
            controlnet_conditioning_scale=_cn_scale,
            num_inference_steps=_steps,
            generator=generator,
        )

        return result.images[0]

    def unload(self) -> None:
        """Release the diffusion pipeline from memory."""
        self._pipe = None
        self._model_loader.unload_component("pipeline")
        self._model_loader.unload_component("controlnet")
