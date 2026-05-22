"""Pure reconstruction launcher: no structural losses, max throughput."""

import argparse
from pathlib import Path

from config import BehaviorGenConfig
from trainer import train

DEFAULT_OUT = Path(__file__).parent / "checkpoints" / "pure_recon"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> BehaviorGenConfig:
    cfg = BehaviorGenConfig(
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_volatility=0.0,
        weight_trend=0.0,
        weight_turning=0.0,
        weight_dtw=0.0,
        weight_smoothness=0.0,
        weight_autocorr=0.0,
        weight_tail=0.0,
        weight_diversity=0.0,
        weight_latent_sensitivity=0.0,
        weight_manifold_spread=0.0,
        training_paths_per_sample=16,
        log_every=10,
        checkpoint_every=1000,
        visualize_every=2500,
    )
    cfg._total_steps = args.steps
    return cfg


if __name__ == "__main__":
    args = parse_args()
    train(output_dir=args.output_dir, config=build_config(args))
