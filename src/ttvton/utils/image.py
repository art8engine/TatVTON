"""Image conversion and resizing utilities."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image


def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """Convert a PIL image to a ``(H, W, C)`` uint8 NumPy array."""
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def numpy_to_pil(array: np.ndarray) -> Image.Image:
    """Convert a ``(H, W, C)`` uint8 array back to PIL RGB."""
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="RGB")


def pil_to_tensor(image: Image.Image, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Convert PIL image to ``(1, C, H, W)`` tensor normalised to ``[0, 1]``."""
    arr = np.array(image.convert("RGB"), dtype=np.uint8)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(dtype)
    return tensor / 255.0


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a ``(1, C, H, W)`` or ``(C, H, W)`` tensor in ``[0, 1]`` to PIL."""
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    arr = (tensor.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr, mode="RGB")


def resize_for_pipeline(
    image: Image.Image,
    target_size: int | tuple[int, int],
    resample: Image.Resampling = Image.Resampling.LANCZOS,
) -> Image.Image:
    """Resize *image* so both dimensions equal *target_size*.

    Parameters:
        image: Source PIL image.
        target_size: Single int for square, or ``(width, height)`` tuple.
        resample: Resampling filter.
    """
    if isinstance(target_size, int):
        target_size = (target_size, target_size)
    return image.resize(target_size, resample=resample)


def extract_alpha_mask(image: Image.Image) -> Image.Image | None:
    """Return the alpha channel as an ``'L'`` mode image, or ``None`` if opaque."""
    if image.mode in ("RGBA", "LA"):
        return image.split()[-1]
    return None
