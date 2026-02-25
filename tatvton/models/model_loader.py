"""Lazy model loading with VRAM management."""

from __future__ import annotations

import gc
import logging
from typing import Any

import torch

from tatvton.config import TatVTONConfig
from tatvton.utils.device import resolve_device, resolve_dtype

logger = logging.getLogger(__name__)


class ModelLoader:
    """Centralised model loading with lazy initialisation and VRAM management.

    Models are loaded on first access and can be individually unloaded
    to reclaim GPU memory between pipeline phases.
    """

    def __init__(self, config: TatVTONConfig) -> None:
        self._config = config
        self._device = resolve_device(config.device)
        self._dtype = resolve_dtype(config.dtype, self._device)
        self._components: dict[str, Any] = {}

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def load_sam_predictor(self) -> Any:
        """Load and return a SAM 2 image predictor."""
        if "sam" not in self._components:
            logger.info("Loading SAM 2 from %s", self._config.sam_model_id)
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            predictor = SAM2ImagePredictor.from_pretrained(
                self._config.sam_model_id,
                device=self._device,
            )
            self._components["sam"] = predictor
        return self._components["sam"]

    def load_controlnet(self) -> Any:
        """Load and return the ControlNet model."""
        if "controlnet" not in self._components:
            logger.info(
                "Loading ControlNet from %s", self._config.controlnet_model_id
            )
            from diffusers import ControlNetModel

            controlnet = ControlNetModel.from_pretrained(
                self._config.controlnet_model_id,
                torch_dtype=self._dtype,
            )
            self._components["controlnet"] = controlnet
        return self._components["controlnet"]

    def load_inpainting_pipeline(self) -> Any:
        """Load and return the SDXL + ControlNet inpainting pipeline.

        This also configures IP-Adapter and applies the chosen offload strategy.
        """
        if "pipeline" not in self._components:
            controlnet = self.load_controlnet()

            logger.info(
                "Loading SDXL inpainting pipeline from %s",
                self._config.sdxl_model_id,
            )
            from diffusers import StableDiffusionXLControlNetInpaintPipeline

            pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
                self._config.sdxl_model_id,
                controlnet=controlnet,
                torch_dtype=self._dtype,
                variant="fp16" if self._dtype == torch.float16 else None,
            )

            logger.info(
                "Loading IP-Adapter: %s / %s",
                self._config.ip_adapter_repo_id,
                self._config.ip_adapter_weight_name,
            )
            pipe.load_ip_adapter(
                self._config.ip_adapter_repo_id,
                subfolder="sdxl_models",
                weight_name=self._config.ip_adapter_weight_name,
            )
            pipe.set_ip_adapter_scale(self._config.ip_adapter_scale)

            self._apply_offload(pipe)
            self._components["pipeline"] = pipe

        return self._components["pipeline"]

    def unload_component(self, name: str) -> None:
        """Unload a named component and free GPU memory."""
        component = self._components.pop(name, None)
        if component is not None:
            del component
            self._flush_gpu()
            logger.info("Unloaded component '%s'", name)

    def unload_all(self) -> None:
        """Unload every loaded component."""
        names = list(self._components.keys())
        self._components.clear()
        self._flush_gpu()
        logger.info("Unloaded all components: %s", names)

    def _apply_offload(self, pipe: Any) -> None:
        """Apply the configured offload strategy to the diffusion pipeline."""
        strategy = self._config.offload_strategy
        if strategy == "sequential":
            pipe.enable_sequential_cpu_offload()
            logger.info("Enabled sequential CPU offload")
        elif strategy == "model":
            pipe.enable_model_cpu_offload()
            logger.info("Enabled model CPU offload")
        else:
            pipe.to(self._device)

    @staticmethod
    def _flush_gpu() -> None:
        """Run garbage collection and clear the CUDA cache."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
