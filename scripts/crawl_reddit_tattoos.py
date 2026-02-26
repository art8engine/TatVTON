"""Crawl tattoo images from Reddit with CLIP quality filter.

Searches by body part + style keywords for diverse coverage.
CLIP verifies "is this a tattoo on skin?" (NOT body part classification).
Accurate body part labeling is deferred to DensePose (GPU, Colab).

No API key required — uses Reddit's public .json endpoints.

Usage:
    pip install requests Pillow torch transformers
    python scripts/crawl_reddit_tattoos.py \
        --output dataset/images \
        --per-category 100 \
        --min-size 512
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import time
from pathlib import Path

import requests
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) TatVTON-Research/1.0"

# -- Search keywords (for Reddit diversity, NOT final labels) ---------------

BODY_KEYWORDS = ["arm", "shoulder", "back", "chest", "neck", "leg"]

STYLE_KEYWORDS = [
    "blackwork",
    "black and grey",
    "traditional",
    "neo traditional",
    "realism",
    "polynesian",
    "japanese",
    "lettering",
    "fine line",
    "watercolor",
    "dotwork",
    "geometric",
]

SUBREDDITS = ["tattoos", "tattoo", "TattooDesigns", "irezumi", "traditionaltattoos"]


# -- CLIP Quality Filter ---------------------------------------------------


class CLIPQualityFilter:
    """Filter images: keep only 'tattoo on human skin' photos.

    Does NOT classify body part (DensePose does that later).
    Only answers: "Is this a photo of a tattoo on a person's body?"
    """

    def __init__(self, device: str = "cpu") -> None:
        model_id = "openai/clip-vit-base-patch32"
        logger.info("Loading CLIP filter: %s on %s", model_id, device)
        self._model = CLIPModel.from_pretrained(model_id).to(device)
        self._processor = CLIPProcessor.from_pretrained(model_id)
        self._device = device

        self._positive_prompts = [
            "a tattoo on human skin",
            "a tattoo on a person's body",
            "a close-up photo of a tattoo on skin",
        ]
        self._negative_prompts = [
            "a tattoo design on paper or white background",
            "a drawing or illustration, not a photo",
            "a photo without any tattoo",
            "a landscape or object photo",
            "text, meme, or screenshot",
        ]
        self._all_prompts = self._positive_prompts + self._negative_prompts

    @torch.no_grad()
    def is_tattoo_on_skin(self, image: Image.Image, threshold: float = 0.5) -> tuple[bool, float]:
        """Check if image shows a tattoo on human skin.

        Returns (is_valid, confidence).
        confidence = sum of positive prompt scores.
        """
        inputs = self._processor(
            text=self._all_prompts, images=image, return_tensors="pt", padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        outputs = self._model(**inputs)
        probs = outputs.logits_per_image[0].softmax(dim=0).cpu().tolist()

        pos_score = sum(probs[:len(self._positive_prompts)])
        neg_score = sum(probs[len(self._positive_prompts):])

        is_valid = pos_score > threshold and pos_score > neg_score
        return is_valid, round(pos_score, 3)

    @torch.no_grad()
    def classify_style(self, image: Image.Image) -> tuple[str, float]:
        """Classify tattoo style (this CLIP can do reasonably well)."""
        prompts = [f"a {s} style tattoo" for s in STYLE_KEYWORDS]
        inputs = self._processor(
            text=prompts, images=image, return_tensors="pt", padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        outputs = self._model(**inputs)
        probs = outputs.logits_per_image[0].softmax(dim=0).cpu().tolist()

        scored = sorted(zip(STYLE_KEYWORDS, probs), key=lambda x: x[1], reverse=True)
        return scored[0][0], round(scored[0][1], 3)


# -- Utility functions -----------------------------------------------------


def _image_hash(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def _is_valid_image_url(url: str) -> bool:
    lower = url.lower()
    return any(lower.endswith(ext) for ext in IMAGE_EXTENSIONS) or "i.redd.it" in lower


def _download_image(url: str, timeout: int = 15) -> bytes | None:
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "")
        if "image" not in content_type and not _is_valid_image_url(url):
            return None
        return resp.content
    except Exception as e:
        logger.warning("Download failed %s: %s", url, e)
        return None


def _check_resolution(data: bytes, min_size: int) -> tuple[int, int] | None:
    try:
        img = Image.open(io.BytesIO(data))
        w, h = img.size
        if min(w, h) >= min_size:
            return w, h
    except Exception:
        pass
    return None


def _save_image(data: bytes, filepath: Path) -> bool:
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        img.save(filepath, "JPEG", quality=95)
        return True
    except Exception as e:
        logger.warning("Save failed %s: %s", filepath.name, e)
        return False


# -- Reddit JSON -----------------------------------------------------------


def _search_reddit_json(
    query: str,
    subreddit: str,
    after: str | None = None,
) -> tuple[list[dict], str | None]:
    url = f"https://www.reddit.com/r/{subreddit}/search.json"
    params: dict[str, str | int] = {
        "q": query, "restrict_sr": 1, "sort": "relevance",
        "t": "all", "limit": 100, "raw_json": 1, "type": "link",
    }
    if after:
        params["after"] = after

    try:
        resp = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=20)
        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", 60))
            logger.warning("Rate limited, waiting %ds...", retry_after)
            time.sleep(retry_after)
            return [], after
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning("Search failed '%s' r/%s: %s", query, subreddit, e)
        return [], None

    posts = [child.get("data", {}) for child in data.get("data", {}).get("children", [])]
    next_after = data.get("data", {}).get("after")
    return posts, next_after


def _get_image_url(post: dict) -> str | None:
    url = post.get("url", "")
    if _is_valid_image_url(url):
        return url
    if post.get("is_gallery") and post.get("media_metadata"):
        for media in post["media_metadata"].values():
            img_url = media.get("s", {}).get("u", "")
            if img_url:
                return img_url.replace("&amp;", "&")
    return None


# -- Core crawl ------------------------------------------------------------


def crawl_by_keywords(
    body_kw: str,
    style_kw: str,
    subreddits: list[str],
    output_dir: Path,
    per_category: int,
    min_size: int,
    seen_hashes: set[str],
    clip_filter: CLIPQualityFilter | None,
) -> list[dict]:
    """Search Reddit, filter with CLIP, save."""
    query = f"{body_kw} {style_kw} tattoo"
    metadata: list[dict] = []
    rejected = 0

    for subreddit in subreddits:
        if len(metadata) >= per_category:
            break

        after = None
        for page in range(10):
            if len(metadata) >= per_category:
                break

            posts, after = _search_reddit_json(query, subreddit, after)
            if not posts:
                break

            for post in posts:
                if len(metadata) >= per_category:
                    break

                img_url = _get_image_url(post)
                if img_url is None:
                    continue

                data = _download_image(img_url)
                if data is None:
                    continue

                img_hash = _image_hash(data)
                if img_hash in seen_hashes:
                    continue
                seen_hashes.add(img_hash)

                resolution = _check_resolution(data, min_size)
                if resolution is None:
                    continue

                pil_img = Image.open(io.BytesIO(data)).convert("RGB")

                # CLIP: is this a tattoo on skin? (not body part classification)
                clip_score = 0.0
                style_label = style_kw
                if clip_filter is not None:
                    is_valid, clip_score = clip_filter.is_tattoo_on_skin(pil_img)
                    if not is_valid:
                        rejected += 1
                        continue
                    style_label, _ = clip_filter.classify_style(pil_img)

                filename = f"{img_hash[:12]}.jpg"
                filepath = output_dir / filename

                if not _save_image(data, filepath):
                    continue

                metadata.append({
                    "filename": filename,
                    "source": f"r/{subreddit}",
                    "reddit_id": post.get("id", ""),
                    "title": post.get("title", ""),
                    "url": img_url,
                    "width": resolution[0],
                    "height": resolution[1],
                    "score": post.get("score", 0),
                    "nsfw": post.get("over_18", False),
                    "search_body": body_kw,
                    "search_style": style_kw,
                    "style_clip": style_label,
                    "clip_score": clip_score,
                    "body_part": None,  # DensePose will fill this later
                })

            time.sleep(5)  # 5s between pages to avoid rate limit
            if after is None:
                break

    if rejected > 0:
        logger.info("  Rejected %d (not tattoo on skin)", rejected)

    return metadata


# -- Main ------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crawl tattoo images from Reddit with CLIP quality filter."
    )
    parser.add_argument("--output", type=Path, default=Path("dataset/images"))
    parser.add_argument("--per-category", type=int, default=100)
    parser.add_argument("--min-size", type=int, default=512)
    parser.add_argument("--no-clip", action="store_true", help="Skip CLIP filtering.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--body-keywords", nargs="+", default=BODY_KEYWORDS)
    parser.add_argument("--style-keywords", nargs="+", default=STYLE_KEYWORDS)
    parser.add_argument("--subreddits", nargs="+", default=SUBREDDITS)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    clip_filter = None
    if not args.no_clip:
        clip_filter = CLIPQualityFilter(device=args.device)

    all_metadata: list[dict] = []
    seen_hashes: set[str] = set()
    total_cats = len(args.body_keywords) * len(args.style_keywords)

    logger.info(
        "Crawling %d body x %d style = %d keyword combos, %d per combo",
        len(args.body_keywords), len(args.style_keywords), total_cats, args.per_category,
    )
    logger.info("CLIP filter: %s (quality only, NOT body part)", "ON" if clip_filter else "OFF")
    logger.info("Body part labeling: deferred to DensePose (Phase 3)")

    cat_idx = 0
    for body_kw in args.body_keywords:
        for style_kw in args.style_keywords:
            cat_idx += 1
            logger.info("[%d/%d] %s + %s ...", cat_idx, total_cats, body_kw, style_kw)

            entries = crawl_by_keywords(
                body_kw=body_kw,
                style_kw=style_kw,
                subreddits=args.subreddits,
                output_dir=args.output,
                per_category=args.per_category,
                min_size=args.min_size,
                seen_hashes=seen_hashes,
                clip_filter=clip_filter,
            )

            all_metadata.extend(entries)
            logger.info("  → %d images (total: %d)", len(entries), len(all_metadata))

            if cat_idx % 10 == 0:
                _save_metadata(all_metadata, args.output.parent)

    _save_metadata(all_metadata, args.output.parent)
    _print_summary(all_metadata)


def _save_metadata(metadata: list[dict], parent_dir: Path) -> None:
    meta_path = parent_dir / "metadata_crawled.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info("Saved: %d entries → %s", len(metadata), meta_path)


def _print_summary(metadata: list[dict]) -> None:
    from collections import Counter

    logger.info("=" * 60)
    logger.info("TOTAL: %d images", len(metadata))

    style_dist = Counter(m["style_clip"] for m in metadata if m.get("style_clip"))
    search_dist = Counter(m["search_body"] for m in metadata)

    logger.info("--- Search Keyword Distribution ---")
    for kw, count in search_dist.most_common():
        logger.info("  %-20s %4d", kw, count)

    logger.info("--- Style (CLIP) Distribution ---")
    for style, count in style_dist.most_common():
        logger.info("  %-25s %4d", style, count)

    logger.info("Body part: all None (to be labeled by DensePose)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
