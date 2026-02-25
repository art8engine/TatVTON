"""Auto-label tattoo images with body part and style/genre.

Uses CLIP for zero-shot classification of:
  1. Body part (arm, leg, back, chest, etc.)
  2. Tattoo style (blackwork, traditional, realism, etc.)

Generates rich text captions for ControlNet training.

Usage:
    pip install torch transformers Pillow
    python scripts/label_dataset.py \
        --input-dir dataset/images \
        --output dataset/metadata_labeled.json \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# -- Classification labels -------------------------------------------------

BODY_PARTS = [
    "arm",
    "shoulder",
    "back",
    "chest",
    "neck",
    "leg",
]

TATTOO_STYLES = [
    "blackwork",
    "black and grey",
    "traditional old school",
    "neo-traditional",
    "realism",
    "polynesian tribal",
    "japanese irezumi",
    "lettering script",
    "minimalist fine line",
    "watercolor",
    "dotwork",
    "geometric",
    "surrealism",
    "chicano",
    "new school",
    "biomechanical",
]

# Readable style names for captions
STYLE_CAPTION_MAP = {
    "blackwork": "blackwork",
    "black and grey": "black and grey",
    "traditional old school": "traditional old school",
    "neo-traditional": "neo-traditional",
    "realism": "realistic",
    "polynesian tribal": "polynesian tribal",
    "japanese irezumi": "japanese irezumi",
    "lettering script": "lettering",
    "minimalist fine line": "minimalist fine line",
    "watercolor": "watercolor",
    "dotwork": "dotwork",
    "geometric": "geometric",
    "surrealism": "surreal",
    "chicano": "chicano",
    "new school": "new school",
    "biomechanical": "biomechanical",
}


class CLIPLabeler:
    """Zero-shot image classifier using CLIP."""

    def __init__(self, device: str = "cuda") -> None:
        from transformers import CLIPModel, CLIPProcessor

        model_id = "openai/clip-vit-large-patch14"
        logger.info("Loading CLIP model: %s", model_id)
        self._model = CLIPModel.from_pretrained(model_id).to(device)
        self._processor = CLIPProcessor.from_pretrained(model_id)
        self._device = device

        # Pre-compute text embeddings for efficiency
        self._body_part_prompts = [f"a tattoo on {part}" for part in BODY_PARTS]
        self._style_prompts = [f"a {style} style tattoo" for style in TATTOO_STYLES]

    @torch.no_grad()
    def classify(
        self, image: Image.Image, labels: list[str], prompts: list[str]
    ) -> tuple[str, float, list[tuple[str, float]]]:
        """Classify image against a set of text prompts.

        Returns:
            (top_label, top_score, all_scores)
        """
        inputs = self._processor(
            text=prompts,
            images=image,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        outputs = self._model(**inputs)
        logits = outputs.logits_per_image[0]
        probs = logits.softmax(dim=0).cpu().tolist()

        scored = list(zip(labels, probs))
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored[0][0], scored[0][1], scored

    def label_body_part(self, image: Image.Image) -> tuple[str, float, list[tuple[str, float]]]:
        """Classify which body part the tattoo is on."""
        return self.classify(image, BODY_PARTS, self._body_part_prompts)

    def label_style(self, image: Image.Image) -> tuple[str, float, list[tuple[str, float]]]:
        """Classify the tattoo style/genre."""
        return self.classify(image, TATTOO_STYLES, self._style_prompts)


def generate_caption(body_part: str, style: str) -> str:
    """Generate a training caption from labels."""
    style_name = STYLE_CAPTION_MAP.get(style, style)
    return f"a photorealistic {style_name} tattoo on {body_part}, high quality, detailed skin texture"


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-label tattoo dataset with CLIP.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory with tattoo images.")
    parser.add_argument("--output", type=Path, default=Path("dataset/metadata_labeled.json"), help="Output metadata.")
    parser.add_argument("--existing-meta", type=Path, default=None, help="Existing metadata to merge with.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"], help="Device.")
    parser.add_argument("--batch-log", type=int, default=50, help="Log every N images.")
    args = parser.parse_args()

    # Load existing metadata if available
    existing: dict[str, dict] = {}
    if args.existing_meta and args.existing_meta.exists():
        with open(args.existing_meta) as f:
            for entry in json.load(f):
                existing[entry["filename"]] = entry
        logger.info("Loaded %d existing metadata entries", len(existing))

    image_paths = sorted(
        list(args.input_dir.glob("*.jpg"))
        + list(args.input_dir.glob("*.jpeg"))
        + list(args.input_dir.glob("*.png"))
    )
    if not image_paths:
        logger.error("No images found in %s", args.input_dir)
        return

    logger.info("Found %d images to label", len(image_paths))

    labeler = CLIPLabeler(device=args.device)
    results: list[dict] = []

    for idx, img_path in enumerate(image_paths):
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.warning("Failed to open %s: %s", img_path.name, e)
            continue

        body_part, body_conf, _ = labeler.label_body_part(img)
        style, style_conf, _ = labeler.label_style(img)
        caption = generate_caption(body_part, style)

        entry = existing.get(img_path.name, {})
        entry.update({
            "filename": img_path.name,
            "body_part": body_part,
            "body_part_confidence": round(body_conf, 3),
            "style": style,
            "style_confidence": round(style_conf, 3),
            "caption": caption,
        })
        results.append(entry)

        if (idx + 1) % args.batch_log == 0:
            logger.info(
                "Labeled %d/%d — last: %s / %s (%.0f%% / %.0f%%)",
                idx + 1, len(image_paths),
                body_part, style,
                body_conf * 100, style_conf * 100,
            )

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info("Labeled %d images → %s", len(results), args.output)

    # Print distribution summary
    from collections import Counter

    body_dist = Counter(r["body_part"] for r in results)
    style_dist = Counter(r["style"] for r in results)

    logger.info("--- Body Part Distribution ---")
    for part, count in body_dist.most_common():
        logger.info("  %-20s %d (%.1f%%)", part, count, count / len(results) * 100)

    logger.info("--- Style Distribution ---")
    for style, count in style_dist.most_common():
        logger.info("  %-25s %d (%.1f%%)", style, count, count / len(results) * 100)


if __name__ == "__main__":
    main()
