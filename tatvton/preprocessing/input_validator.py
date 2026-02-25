"""Input validation for pipeline arguments."""

from __future__ import annotations

from PIL import Image

from tatvton.types import BBoxPrompt, PointPrompt, RegionPrompt


class InputValidator:
    """Validates inputs before they enter the pipeline."""

    MIN_IMAGE_SIZE = 64

    @staticmethod
    def validate_image(image: Image.Image, name: str = "image") -> None:
        """Ensure *image* is a valid PIL image with minimum dimensions."""
        if not isinstance(image, Image.Image):
            raise TypeError(f"{name} must be a PIL Image, got {type(image).__name__}")
        w, h = image.size
        if w < InputValidator.MIN_IMAGE_SIZE or h < InputValidator.MIN_IMAGE_SIZE:
            raise ValueError(
                f"{name} too small: {w}x{h}. "
                f"Minimum is {InputValidator.MIN_IMAGE_SIZE}x{InputValidator.MIN_IMAGE_SIZE}."
            )

    @staticmethod
    def validate_region(region: RegionPrompt, image: Image.Image) -> None:
        """Ensure *region* coordinates lie within *image* bounds."""
        w, h = image.size

        if isinstance(region, PointPrompt):
            for x, y in region.coords:
                if not (0 <= x < w and 0 <= y < h):
                    raise ValueError(
                        f"Point ({x}, {y}) is outside image bounds ({w}x{h})."
                    )
        elif isinstance(region, BBoxPrompt):
            x_min, y_min, x_max, y_max = region.bbox
            if x_min < 0 or y_min < 0 or x_max > w or y_max > h:
                raise ValueError(
                    f"BBox {region.bbox} is outside image bounds ({w}x{h})."
                )
        else:
            raise TypeError(
                f"region must be PointPrompt or BBoxPrompt, got {type(region).__name__}"
            )

    @staticmethod
    def validate_call(
        body_image: Image.Image,
        tattoo_image: Image.Image,
        region: RegionPrompt,
        strength: float | None = None,
        ip_adapter_scale: float | None = None,
        num_inference_steps: int | None = None,
    ) -> None:
        """Run all input validations for a pipeline call."""
        InputValidator.validate_image(body_image, "body_image")
        InputValidator.validate_image(tattoo_image, "tattoo_image")
        InputValidator.validate_region(region, body_image)

        if strength is not None and not 0.0 < strength <= 1.0:
            raise ValueError(f"strength must be in (0.0, 1.0], got {strength}")
        if ip_adapter_scale is not None and not 0.0 <= ip_adapter_scale <= 2.0:
            raise ValueError(
                f"ip_adapter_scale must be in [0.0, 2.0], got {ip_adapter_scale}"
            )
        if num_inference_steps is not None and num_inference_steps < 1:
            raise ValueError(
                f"num_inference_steps must be >= 1, got {num_inference_steps}"
            )
