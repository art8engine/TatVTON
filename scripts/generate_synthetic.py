"""Generate synthetic tattoo-on-body paired data.

Takes clean body images + tattoo designs → composites them together.
Produces paired (with_tattoo, without_tattoo) for ControlNet training.

Usage:
    pip install Pillow numpy datasets
    python scripts/generate_synthetic.py \
        --body-dir dataset/images \
        --output-dir dataset/synthetic \
        --num-samples 5000
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Body part keywords for caption generation
BODY_PARTS = [
    "arm", "shoulder", "back", "chest", "neck", "leg",
]

TATTOO_STYLES = [
    "realistic", "traditional", "geometric", "minimalist",
    "watercolor", "tribal", "japanese", "blackwork",
    "fine line", "dotwork", "neo-traditional", "illustrative",
]


def load_tattoo_designs(source: str = "huggingface") -> list[Image.Image]:
    """Load tattoo designs from Drozdik/tattoo_v3 or a local directory."""
    designs = []

    if source == "huggingface":
        try:
            from datasets import load_dataset

            logger.info("Loading Drozdik/tattoo_v3 from HuggingFace...")
            ds = load_dataset("Drozdik/tattoo_v3", split="train")
            for row in ds:
                designs.append(row["image"].convert("RGBA"))
            logger.info("Loaded %d tattoo designs from HF", len(designs))
        except Exception as e:
            logger.error("Failed to load from HuggingFace: %s", e)
            raise
    else:
        design_dir = Path(source)
        for p in sorted(design_dir.glob("*.png")) + sorted(design_dir.glob("*.jpg")):
            designs.append(Image.open(p).convert("RGBA"))
        logger.info("Loaded %d tattoo designs from %s", len(designs), design_dir)

    return designs


def remove_white_background(img: Image.Image, threshold: int = 240) -> Image.Image:
    """Remove white background from tattoo design, making it transparent."""
    arr = np.array(img)
    if arr.shape[2] == 3:
        alpha = np.ones((*arr.shape[:2], 1), dtype=np.uint8) * 255
        arr = np.concatenate([arr, alpha], axis=2)

    # White pixels become transparent
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    white_mask = (r > threshold) & (g > threshold) & (b > threshold)
    arr[white_mask, 3] = 0

    # Smooth alpha edges
    result = Image.fromarray(arr, "RGBA")
    alpha_channel = result.split()[3]
    alpha_channel = alpha_channel.filter(ImageFilter.GaussianBlur(radius=1))
    result.putalpha(alpha_channel)

    return result


def random_perspective_transform(
    tattoo: Image.Image,
    target_size: tuple[int, int],
    scale_range: tuple[float, float] = (0.15, 0.45),
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Apply random perspective warp and scale to tattoo design.

    Returns the transformed tattoo and its bounding box (x1, y1, x2, y2).
    """
    tw, th = target_size

    # Random scale
    scale = random.uniform(*scale_range)
    new_w = int(tw * scale)
    new_h = int(new_w * tattoo.height / tattoo.width)

    if new_h > th * 0.6:
        new_h = int(th * 0.6)
        new_w = int(new_h * tattoo.width / tattoo.height)

    tattoo_resized = tattoo.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Random slight rotation
    angle = random.uniform(-15, 15)
    tattoo_rotated = tattoo_resized.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)

    # Random position (ensure it fits)
    rw, rh = tattoo_rotated.size
    max_x = max(tw - rw, 1)
    max_y = max(th - rh, 1)
    x = random.randint(0, max_x)
    y = random.randint(0, max_y)

    # Create full-size RGBA canvas
    canvas = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
    canvas.paste(tattoo_rotated, (x, y), tattoo_rotated)

    bbox = (x, y, min(x + rw, tw), min(y + rh, th))
    return canvas, bbox


def apply_skin_blend(
    body: Image.Image,
    tattoo_overlay: Image.Image,
    opacity: float = 0.85,
) -> Image.Image:
    """Blend tattoo onto body with skin-like opacity for realism."""
    body_rgba = body.convert("RGBA")

    # Reduce tattoo opacity for more realistic skin blend
    tattoo_arr = np.array(tattoo_overlay).astype(np.float32)
    tattoo_arr[:, :, 3] *= opacity
    tattoo_adjusted = Image.fromarray(tattoo_arr.astype(np.uint8), "RGBA")

    # Composite
    result = Image.alpha_composite(body_rgba, tattoo_adjusted)
    return result.convert("RGB")


def generate_caption(bbox: tuple[int, int, int, int], img_size: tuple[int, int]) -> str:
    """Generate a text caption based on tattoo placement."""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2 / img_size[0]
    cy = (y1 + y2) / 2 / img_size[1]

    # Rough body part estimation based on position
    if cy < 0.3:
        part = random.choice(["shoulder", "neck", "chest"])
    elif cy < 0.6:
        part = random.choice(["arm", "back", "chest"])
    else:
        part = random.choice(["leg", "arm"])

    style = random.choice(TATTOO_STYLES)
    return f"a photorealistic {style} tattoo on {part}, high quality, detailed skin texture"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic tattoo pairs.")
    parser.add_argument("--body-dir", type=Path, required=True, help="Directory with body/skin images.")
    parser.add_argument("--tattoo-source", default="huggingface", help="'huggingface' or path to tattoo designs dir.")
    parser.add_argument("--output-dir", type=Path, default=Path("dataset/synthetic"), help="Output directory.")
    parser.add_argument("--num-samples", type=int, default=5000, help="Number of pairs to generate.")
    parser.add_argument("--resolution", type=int, default=1024, help="Output resolution.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Create output directories
    images_dir = args.output_dir / "images"       # with tattoo (target)
    clean_dir = args.output_dir / "clean"          # without tattoo (for reference)
    images_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)

    # Load inputs
    logger.info("Loading tattoo designs...")
    tattoo_designs = load_tattoo_designs(args.tattoo_source)
    tattoo_designs = [remove_white_background(t) for t in tattoo_designs]
    logger.info("Processed %d tattoo designs (bg removed)", len(tattoo_designs))

    body_paths = sorted(
        list(args.body_dir.glob("*.jpg"))
        + list(args.body_dir.glob("*.jpeg"))
        + list(args.body_dir.glob("*.png"))
    )
    if not body_paths:
        logger.error("No body images found in %s", args.body_dir)
        return
    logger.info("Found %d body images", len(body_paths))

    res = args.resolution
    metadata = []

    for i in range(args.num_samples):
        body_path = random.choice(body_paths)
        tattoo_design = random.choice(tattoo_designs)

        try:
            body = Image.open(body_path).convert("RGB")
            body_resized = body.resize((res, res), Image.Resampling.LANCZOS)
        except Exception as e:
            logger.warning("Failed to load %s: %s", body_path, e)
            continue

        tattoo_overlay, bbox = random_perspective_transform(
            tattoo_design, (res, res)
        )
        result = apply_skin_blend(body_resized, tattoo_overlay)
        caption = generate_caption(bbox, (res, res))

        filename = f"{i:06d}.jpg"
        result.save(images_dir / filename, "JPEG", quality=95)
        body_resized.save(clean_dir / filename, "JPEG", quality=95)

        metadata.append({
            "filename": filename,
            "caption": caption,
            "bbox": list(bbox),
            "source_body": body_path.name,
            "synthetic": True,
        })

        if (i + 1) % 100 == 0:
            logger.info("Generated %d / %d pairs", i + 1, args.num_samples)

    meta_path = args.output_dir / "metadata_synthetic.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info("Done: %d pairs saved to %s", len(metadata), args.output_dir)


if __name__ == "__main__":
    main()
