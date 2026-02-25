"""Hugging Face Hub download and cache helpers."""

from __future__ import annotations

import logging
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

logger = logging.getLogger(__name__)


def download_model(
    repo_id: str,
    filename: str | None = None,
    cache_dir: str | None = None,
    revision: str | None = None,
) -> Path:
    """Download a single file from the Hub and return its local path.

    Args:
        repo_id: HF Hub repository id (e.g. ``"h94/IP-Adapter"``).
        filename: File within the repo to download.
        cache_dir: Local cache directory override.
        revision: Git revision (branch / tag / commit SHA).

    Returns:
        :class:`Path` to the downloaded file.
    """
    if filename:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir,
            revision=revision,
        )
    else:
        path = snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            revision=revision,
        )
    logger.info("Downloaded %s/%s → %s", repo_id, filename or "(snapshot)", path)
    return Path(path)
