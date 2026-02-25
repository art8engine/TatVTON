"""ControlNet SDXL training launcher for TatVTON.

Wrapper around diffusers' official train_controlnet_sdxl.py with
TatVTON-specific defaults.

Usage (Colab / A100):
    python scripts/train_controlnet.py \
        --dataset-id khope/tatvton-dataset-v1 \
        --output-dir ./checkpoints/tatvton-controlnet-v1 \
        --epochs 3

    # After training, push to HF Hub:
    python scripts/train_controlnet.py --push-model ./checkpoints/tatvton-controlnet-v1
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DIFFUSERS_TRAIN_SCRIPT = "diffusers/examples/controlnet/train_controlnet_sdxl.py"


def ensure_diffusers_repo() -> Path:
    """Clone diffusers repo if not present."""
    diffusers_dir = Path("diffusers")
    if not diffusers_dir.exists():
        logger.info("Cloning diffusers repository...")
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/huggingface/diffusers.git"],
            check=True,
        )
    train_script = Path(DIFFUSERS_TRAIN_SCRIPT)
    if not train_script.exists():
        logger.error("Training script not found: %s", train_script)
        sys.exit(1)
    return train_script


def install_training_deps() -> None:
    """Install training dependencies."""
    reqs = Path("diffusers/examples/controlnet/requirements_sdxl.txt")
    if reqs.exists():
        logger.info("Installing training requirements...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(reqs)], check=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "accelerate", "wandb", "datasets"],
        check=True,
    )


def push_model(model_dir: str, repo_id: str = "khope/tatvton-controlnet-v1") -> None:
    """Push trained ControlNet to HuggingFace Hub."""
    from diffusers import ControlNetModel

    logger.info("Loading model from %s", model_dir)
    model = ControlNetModel.from_pretrained(model_dir)
    logger.info("Pushing to %s", repo_id)
    model.push_to_hub(repo_id)
    logger.info("Done! Model available at https://huggingface.co/%s", repo_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ControlNet SDXL for TatVTON.")

    # Mode
    parser.add_argument("--push-model", type=str, help="Push trained model dir to HF Hub (skip training).")
    parser.add_argument("--hub-model-id", default="khope/tatvton-controlnet-v1", help="HF Hub model ID.")

    # Dataset
    parser.add_argument("--dataset-id", default="khope/tatvton-dataset-v1", help="HF dataset ID.")
    parser.add_argument("--image-column", default="image", help="Dataset image column.")
    parser.add_argument("--conditioning-column", default="conditioning_image", help="Dataset conditioning column.")
    parser.add_argument("--caption-column", default="text", help="Dataset text column.")

    # Model
    parser.add_argument(
        "--base-model",
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Base SDXL model.",
    )

    # Training
    parser.add_argument("--output-dir", type=Path, default=Path("./checkpoints/tatvton-controlnet-v1"))
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=1, help="Train batch size.")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--resolution", type=int, default=1024, help="Training resolution.")
    parser.add_argument("--mixed-precision", default="fp16", choices=["fp16", "bf16", "no"])
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Enable for lower VRAM.")
    parser.add_argument("--save-steps", type=int, default=500, help="Checkpoint save interval.")
    parser.add_argument("--validation-steps", type=int, default=200, help="Validation interval.")
    parser.add_argument("--seed", type=int, default=42)

    # Logging
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging.")
    parser.add_argument("--wandb-project", default="tatvton-controlnet")

    args = parser.parse_args()

    # Push mode
    if args.push_model:
        push_model(args.push_model, args.hub_model_id)
        return

    # Training mode
    train_script = ensure_diffusers_repo()
    install_training_deps()

    cmd = [
        "accelerate", "launch", str(train_script),
        f"--pretrained_model_name_or_path={args.base_model}",
        f"--output_dir={args.output_dir}",
        f"--dataset_name={args.dataset_id}",
        f"--image_column={args.image_column}",
        f"--conditioning_image_column={args.conditioning_column}",
        f"--caption_column={args.caption_column}",
        f"--resolution={args.resolution}",
        f"--learning_rate={args.lr}",
        f"--train_batch_size={args.batch_size}",
        f"--gradient_accumulation_steps={args.grad_accum}",
        f"--num_train_epochs={args.epochs}",
        f"--mixed_precision={args.mixed_precision}",
        f"--checkpointing_steps={args.save_steps}",
        f"--validation_steps={args.validation_steps}",
        f"--seed={args.seed}",
    ]

    if args.gradient_checkpointing:
        cmd.append("--enable_xformers_memory_efficient_attention")
        cmd.append("--gradient_checkpointing")

    if args.wandb:
        cmd.extend([
            "--report_to=wandb",
            f"--tracker_project_name={args.wandb_project}",
        ])

    logger.info("Training command:\n  %s", " \\\n    ".join(cmd))
    logger.info("Starting training...")

    result = subprocess.run(cmd)

    if result.returncode == 0:
        logger.info("Training complete! Model saved to %s", args.output_dir)
        logger.info("To push: python scripts/train_controlnet.py --push-model %s", args.output_dir)
    else:
        logger.error("Training failed with exit code %d", result.returncode)
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
