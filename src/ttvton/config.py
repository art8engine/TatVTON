"""Pipeline configuration — frozen dataclass with sensible defaults."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True)
class TatVTONConfig:
    """Immutable configuration for :class:`~ttvton.TatVTONPipeline`.

    Three-level override strategy:
      1. Defaults here (optimised for 12 GB GPU).
      2. Pass a custom ``TatVTONConfig`` to the pipeline constructor.
      3. Per-call keyword overrides in ``pipeline.__call__(...)``.
    """

    # -- Model IDs --------------------------------------------------------
    sdxl_model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"
    controlnet_model_id: str = "khope/tatvton-controlnet-v1"
    ip_adapter_repo_id: str = "h94/IP-Adapter"
    ip_adapter_weight_name: str = "ip-adapter-plus_sdxl_vit-h.safetensors"
    sam_model_id: str = "facebook/sam2-hiera-large"

    # -- Inference parameters ---------------------------------------------
    resolution: int = 1024
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    strength: float = 0.85
    controlnet_conditioning_scale: float = 0.5
    ip_adapter_scale: float = 0.6

    # -- Mask processing --------------------------------------------------
    mask_dilation_pixels: int = 10
    mask_feather_sigma: float = 5.0

    # -- Device / precision -----------------------------------------------
    device: str = "cuda"
    dtype: torch.dtype = torch.float16
    offload_strategy: Literal["none", "model", "sequential"] = "model"

    # -- Optional features ------------------------------------------------
    use_densepose: bool = False
    seed: int | None = None

    def replace(self, **overrides: object) -> TatVTONConfig:
        """Return a new config with the given fields replaced."""
        from dataclasses import asdict

        current = asdict(self)
        current.update(overrides)
        return TatVTONConfig(**current)
