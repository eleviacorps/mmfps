"""Quick forward-pass debug script."""

import argparse

import torch

from config import BehaviorGenConfig
from generator import BehaviorDiffusionGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-channels", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-paths", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = BehaviorGenConfig()
    if args.base_channels is not None:
        cfg.base_channels = args.base_channels
    cfg.batch_size = args.batch_size

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Creating model...")
    model = BehaviorDiffusionGenerator(cfg).to(device)
    print(f"Model created. Params: {model.count_parameters():,}")
    print(f"base_channels={cfg.base_channels}")

    short = torch.randn(args.batch_size, cfg.short_horizon, cfg.feature_dim).to(device)
    mid = torch.randn(args.batch_size, cfg.mid_horizon, cfg.feature_dim).to(device)
    long = torch.randn(args.batch_size, cfg.long_horizon, cfg.feature_dim).to(device)

    print("Running forward...")

    with torch.no_grad():
        out = model.generate(
            short_seq=short,
            mid_seq=mid,
            long_seq=long,
            num_paths=args.num_paths,
        )

    print("SUCCESS")
    print(f"Output paths: {out[0].shape}")


if __name__ == "__main__":
    main()
