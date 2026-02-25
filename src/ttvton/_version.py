"""Version resolution via importlib.metadata (set by hatch-vcs at build time)."""

from __future__ import annotations


def _get_version() -> str:
    try:
        from importlib.metadata import version

        return version("ttvton")
    except Exception:
        return "0.0.0"


__version__ = _get_version()
