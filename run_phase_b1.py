"""Phase B1: tiny structural refinement after pure diffusion emergence.

By default this starts from the pure reconstruction checkpoint as weights only,
with a fresh optimizer and LR schedule. That avoids inheriting Stage A's final
scheduler state. CLI overrides make short continuation probes reproducible.
"""

import argparse
from pathlib import Path

from config import BehaviorGenConfig
from trainer import train

ROOT = Path(__file__).parent
DEFAULT_OUT = ROOT / "checkpoints" / "phase_b1"
DEFAULT_RESUME = ROOT / "checkpoints" / "pure_recon" / "step_5000_final.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", default=str(DEFAULT_RESUME), help="Checkpoint to load.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT), help="Checkpoint/log output directory.")
    parser.add_argument("--steps", type=int, default=1500, help="Training steps for this run.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--training-paths-per-sample", type=int, default=16)
    parser.add_argument("--vol-weight", type=float, default=0.03)
    parser.add_argument("--turning-weight", type=float, default=0.01)
    parser.add_argument(
        "--full-resume",
        action="store_true",
        help="Resume optimizer/scheduler/global step too. Default reloads weights only.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> BehaviorGenConfig:
    cfg = BehaviorGenConfig(
        batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        weight_volatility=args.vol_weight,
        weight_trend=0.0,
        weight_turning=args.turning_weight,
        weight_dtw=0.0,
        weight_smoothness=0.0,
        weight_autocorr=0.0,
        weight_tail=0.0,
        weight_diversity=0.0,
        weight_latent_sensitivity=0.0,
        weight_manifold_spread=0.0,
        training_paths_per_sample=args.training_paths_per_sample,
        log_every=10,
        checkpoint_every=500,
        visualize_every=1000,
    )
    cfg._total_steps = args.steps
    return cfg


if __name__ == "__main__":
    args = parse_args()
    train(
        output_dir=args.output_dir,
        config=build_config(args),
        resume_from=args.resume,
        resume_weights_only=not args.full_resume,
    )
