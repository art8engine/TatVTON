"""Optional dependency guards.

Usage::

    from tatvton.utils.imports import require_densepose

    require_densepose()  # raises ImportError with install hint if missing
"""

from __future__ import annotations

import importlib
from functools import lru_cache


@lru_cache(maxsize=1)
def is_densepose_available() -> bool:
    """Return ``True`` if detectron2 (DensePose) is importable."""
    try:
        importlib.import_module("detectron2")
        return True
    except ImportError:
        return False


@lru_cache(maxsize=1)
def is_sam2_available() -> bool:
    """Return ``True`` if SAM 2 is importable."""
    try:
        importlib.import_module("sam2")
        return True
    except ImportError:
        return False


def require_densepose() -> None:
    """Raise :class:`ImportError` with install instructions if DensePose is missing."""
    if not is_densepose_available():
        raise ImportError(
            "DensePose requires detectron2. "
            'Install with: pip install "tatvton[densepose]"'
        )


def require_sam2() -> None:
    """Raise :class:`ImportError` if SAM 2 is missing."""
    if not is_sam2_available():
        raise ImportError(
            "SAM 2 is required but not installed. "
            "Install with: pip install segment-anything-2"
        )
