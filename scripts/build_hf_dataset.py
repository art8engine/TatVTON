"""Build HuggingFace Dataset from processed images and upload to Hub.

Expects the following directory structure:
    dataset/
    ├── images/          # Target images (tattoo on body)
    ├── conditioning/    # DensePose IUV maps (same filenames as images, .png)
    └── metadata.json    # Or builds from metadata_crawled.json + metadata_synthetic.json

Usage:
    pip install datasets Pillow huggingface-hub
    python scripts/build_hf_dataset.py \
        --image-dir dataset/images \
        --conditioning-dir dataset/conditioning \
        --repo-id khope/tatvton-dataset-v1 \
        --push
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CAPTION = "a photorealistic tattoo on skin, high quality, detailed"


def load_metadata(dataset_dir: Path) -> dict[str, str]:
    """Load captions from available metadata files.

    Returns a mapping of filename -> caption text.
    """
    captions: dict[str, str] = {}

    for meta_name in ["metadata_crawled.json", "metadata_synthetic.json", "metadata.json"]:
        meta_path = dataset_dir / meta_name
        if meta_path.exists():
            with open(meta_path) as f:
                entries = json.load(f)
            for entry in entries:
                fname = entry.get("filename", "")
                caption = entry.get("caption") or entry.get("title") or DEFAULT_CAPTION
                if fname:
                    captions[fname] = caption
            logger.info("Loaded %d captions from %s", len(entries), meta_name)

    return captions


def build_dataset(
    image_dir: Path,
    conditioning_dir: Path,
    captions: dict[str, str],
) -> dict[str, list]:
    """Build dataset dict from image/conditioning pairs."""
    data: dict[str, list] = {
        "image": [],
        "conditioning_image": [],
        "text": [],
    }

    image_paths = sorted(
        list(image_dir.glob("*.jpg"))
        + list(image_dir.glob("*.jpeg"))
        + list(image_dir.glob("*.png"))
    )

    matched = 0
    skipped = 0

    for img_path in image_paths:
        # Find matching conditioning image (same stem, .png)
        cond_path = conditioning_dir / f"{img_path.stem}.png"
        if not cond_path.exists():
            skipped += 1
            continue

        # Validate both images
        try:
            img = Image.open(img_path).convert("RGB")
            cond = Image.open(cond_path).convert("RGB")
        except Exception as e:
            logger.warning("Failed to open %s or its conditioning: %s", img_path.name, e)
            skipped += 1
            continue

        caption = captions.get(img_path.name, DEFAULT_CAPTION)

        data["image"].append(img)
        data["conditioning_image"].append(cond)
        data["text"].append(caption)
        matched += 1

    logger.info("Matched %d pairs, skipped %d images (no conditioning)", matched, skipped)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Build HF dataset from image/conditioning pairs.")
    parser.add_argument("--image-dir", type=Path, required=True, help="Directory with target images.")
    parser.add_argument("--conditioning-dir", type=Path, required=True, help="Directory with IUV conditioning maps.")
    parser.add_argument("--dataset-dir", type=Path, default=None, help="Parent dataset dir for metadata (default: image-dir parent).")
    parser.add_argument("--output-dir", type=Path, default=Path("dataset/hf_dataset"), help="Local save directory.")
    parser.add_argument("--repo-id", default="khope/tatvton-dataset-v1", help="HF Hub repo ID.")
    parser.add_argument("--push", action="store_true", help="Push to HuggingFace Hub.")
    parser.add_argument("--private", action="store_true", help="Make the HF dataset private.")
    parser.add_argument("--train-split", type=float, default=0.9, help="Train/validation split ratio.")
    args = parser.parse_args()

    try:
        from datasets import Dataset, DatasetDict, Features, Value
        from datasets import Image as HFImage
    except ImportError:
        logger.error("Install datasets: pip install datasets")
        return

    dataset_dir = args.dataset_dir or args.image_dir.parent
    captions = load_metadata(dataset_dir)
    logger.info("Total captions loaded: %d", len(captions))

    data = build_dataset(args.image_dir, args.conditioning_dir, captions)
    total = len(data["image"])

    if total == 0:
        logger.error("No valid pairs found. Check your directories.")
        return

    logger.info("Building HF Dataset with %d samples...", total)

    features = Features({
        "image": HFImage(),
        "conditioning_image": HFImage(),
        "text": Value("string"),
    })

    dataset = Dataset.from_dict(data, features=features)

    # Train/validation split
    split_idx = int(total * args.train_split)
    ds_dict = DatasetDict({
        "train": Dataset.from_dict({
            k: v[:split_idx] for k, v in data.items()
        }, features=features),
        "validation": Dataset.from_dict({
            k: v[split_idx:] for k, v in data.items()
        }, features=features),
    })

    logger.info("Train: %d, Validation: %d", len(ds_dict["train"]), len(ds_dict["validation"]))

    # Save locally
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ds_dict.save_to_disk(str(args.output_dir))
    logger.info("Saved to %s", args.output_dir)

    # Push to Hub
    if args.push:
        logger.info("Pushing to HuggingFace Hub: %s", args.repo_id)
        ds_dict.push_to_hub(args.repo_id, private=args.private)
        logger.info("Uploaded to https://huggingface.co/datasets/%s", args.repo_id)
    else:
        logger.info("Skipping push (use --push to upload to HF Hub)")

    logger.info("Done!")


if __name__ == "__main__":
    main()
