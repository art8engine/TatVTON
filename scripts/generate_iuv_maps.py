"""Batch DensePose IUV map generation for ControlNet conditioning.

Runs DensePose on all images in a directory and saves IUV maps.
Requires GPU + detectron2 (best run on Colab).

Usage:
    pip install 'git+https://github.com/facebookresearch/detectron2.git'
    python scripts/generate_iuv_maps.py \
        --input-dir dataset/images \
        --output-dir dataset/conditioning \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# DensePose model config
DENSEPOSE_CONFIG = "detectron2/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml"
DENSEPOSE_WEIGHTS = (
    "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl"
)


def build_densepose_predictor(device: str = "cuda"):
    """Build a DensePose predictor using detectron2."""
    try:
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor
    except ImportError:
        logger.error(
            "detectron2 not installed. Run:\n"
            "  pip install 'git+https://github.com/facebookresearch/detectron2.git'"
        )
        sys.exit(1)

    # Try to get DensePose config
    try:
        from densepose import add_densepose_config
    except ImportError:
        logger.error(
            "DensePose not available. Install detectron2 with DensePose:\n"
            "  pip install 'git+https://github.com/facebookresearch/detectron2.git'\n"
            "  pip install 'git+https://github.com/facebookresearch/detectron2@main#subdirectory=projects/DensePose'"
        )
        sys.exit(1)

    cfg = get_cfg()
    add_densepose_config(cfg)
    cfg.merge_from_file(DENSEPOSE_CONFIG)
    cfg.MODEL.WEIGHTS = DENSEPOSE_WEIGHTS
    cfg.MODEL.DEVICE = device
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    return DefaultPredictor(cfg)


def extract_iuv(predictor, image: np.ndarray) -> np.ndarray:
    """Run DensePose and extract IUV map.

    Args:
        predictor: detectron2 DefaultPredictor with DensePose.
        image: (H, W, 3) BGR uint8 numpy array.

    Returns:
        (H, W, 3) uint8 IUV map where:
          - channel 0 (I): body part index (0-24)
          - channel 1 (U): U coordinate (0-255)
          - channel 2 (V): V coordinate (0-255)
    """
    h, w = image.shape[:2]
    iuv = np.zeros((h, w, 3), dtype=np.uint8)

    outputs = predictor(image)
    instances = outputs["instances"]

    if not instances.has("pred_densepose"):
        logger.warning("No person detected, returning empty IUV map")
        return iuv

    dp = instances.pred_densepose
    boxes = instances.pred_boxes.tensor.cpu().numpy()

    for i in range(len(dp)):
        x1, y1, x2, y2 = boxes[i].astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        bh, bw = y2 - y1, x2 - x1
        if bh <= 0 or bw <= 0:
            continue

        result_i = dp[i].labels.cpu().numpy().astype(np.uint8)
        result_u = (dp[i].uv[0].cpu().numpy() * 255).astype(np.uint8)
        result_v = (dp[i].uv[1].cpu().numpy() * 255).astype(np.uint8)

        # Resize to bounding box size
        i_resized = np.array(Image.fromarray(result_i).resize((bw, bh), Image.Resampling.NEAREST))
        u_resized = np.array(Image.fromarray(result_u).resize((bw, bh), Image.Resampling.BILINEAR))
        v_resized = np.array(Image.fromarray(result_v).resize((bw, bh), Image.Resampling.BILINEAR))

        iuv[y1:y2, x1:x2, 0] = i_resized
        iuv[y1:y2, x1:x2, 1] = u_resized
        iuv[y1:y2, x1:x2, 2] = v_resized

    return iuv


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch DensePose IUV map generation.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory with body images.")
    parser.add_argument("--output-dir", type=Path, default=Path("dataset/conditioning"), help="Output IUV directory.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device (default: cuda).")
    parser.add_argument("--skip-existing", action="store_true", help="Skip images that already have IUV maps.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        list(args.input_dir.glob("*.jpg"))
        + list(args.input_dir.glob("*.jpeg"))
        + list(args.input_dir.glob("*.png"))
    )
    if not image_paths:
        logger.error("No images found in %s", args.input_dir)
        sys.exit(1)

    logger.info("Found %d images in %s", len(image_paths), args.input_dir)
    logger.info("Building DensePose predictor on %s...", args.device)
    predictor = build_densepose_predictor(args.device)

    results = []
    skipped = 0
    failed = 0

    for idx, img_path in enumerate(image_paths):
        out_path = args.output_dir / f"{img_path.stem}.png"

        if args.skip_existing and out_path.exists():
            skipped += 1
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            img_bgr = np.array(img)[:, :, ::-1]  # RGB → BGR for detectron2

            iuv = extract_iuv(predictor, img_bgr)

            # Check if person was detected (non-zero IUV)
            has_person = iuv.sum() > 0

            Image.fromarray(iuv, "RGB").save(out_path)

            results.append({
                "source": img_path.name,
                "iuv": out_path.name,
                "has_person": bool(has_person),
                "width": img.width,
                "height": img.height,
            })
        except Exception as e:
            logger.warning("Failed on %s: %s", img_path.name, e)
            failed += 1
            continue

        if (idx + 1) % 50 == 0:
            logger.info(
                "Progress: %d/%d (skipped=%d, failed=%d)",
                idx + 1, len(image_paths), skipped, failed,
            )

    # Save metadata
    meta_path = args.output_dir.parent / "metadata_iuv.json"
    with open(meta_path, "w") as f:
        json.dump(results, f, indent=2)

    total_ok = len(results)
    has_person_count = sum(1 for r in results if r["has_person"])

    logger.info("Done: %d IUV maps generated (%d with person detected)", total_ok, has_person_count)
    logger.info("Skipped: %d, Failed: %d", skipped, failed)
    logger.info("Metadata: %s", meta_path)


if __name__ == "__main__":
    main()
