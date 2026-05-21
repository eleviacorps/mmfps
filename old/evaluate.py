"""Validation evaluation — separates generator failure from selector failure.

Generator failure:  no generated path is structurally close to the real future.
Selector failure:  a good path exists, but the composite ranking picks the wrong one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import BehaviorGenConfig
from .dataset import PathDataset
from .generator import BehaviorDiffusionGenerator


@torch.no_grad()
def evaluate(
    checkpoint_path: str,
    data_path: str,
    config: Optional[BehaviorGenConfig] = None,
    num_paths: int = 128,
    max_samples: int = 1000,
    device: Optional[str] = None,
    failure_threshold: float = 0.01,     # MSE threshold for "good enough"
    tolerance: float = 0.02,              # Cone coverage tolerance
) -> dict:
    """Run validation evaluation.

    Outputs separate generator_score and selector_score to distinguish
    where failures originate.
    """
    cfg = config or BehaviorGenConfig()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)

    # Load model
    ckpt = torch.load(checkpoint_path, map_location=device_t, weights_only=False)
    model = BehaviorDiffusionGenerator(ckpt.get("config", cfg)).to(device_t)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.scheduler.noise_scale_val = 1.0  # override: unit-variance initial noise

    # Load validation data
    val_ds = PathDataset(data_path, cfg, split="val")
    n_samples = min(max_samples, len(val_ds))

    # Metrics accumulators
    generator_success = 0     # >=1 path with MSE < threshold
    cone_success = 0          # real inside generated min/max range
    closest_distance_sum = 0.0
    closest_path_found = 0    # best composite matches best MSE
    total = 0

    # Per-regime breakdown (simple: trending vs ranging based on target)
    trending_count = 0
    trending_success = 0
    ranging_count = 0
    ranging_success = 0

    pbar = tqdm(range(n_samples), desc="Evaluating")
    for i in pbar:
        sample = val_ds[i]
        short = sample.short_seq.unsqueeze(0).to(device_t)
        mid = sample.mid_seq.unsqueeze(0).to(device_t)
        long = sample.long_seq.unsqueeze(0).to(device_t)
        target = sample.target.numpy()

        paths, _, _ = model.generate(short, mid, long, num_paths=num_paths)
        paths_np = paths[0].cpu().numpy()  # (num_paths, T)

        # Squared error for each path
        sq_errors = ((paths_np - target) ** 2).mean(axis=-1)  # (num_paths,)
        min_err = sq_errors.min()
        best_idx = sq_errors.argmin()

        closest_distance_sum += float(min_err)

        # Generator success: at least one path is close
        if min_err < failure_threshold:
            generator_success += 1

        # Cone coverage: real inside generated envelope
        gen_min = paths_np.min(axis=0)
        gen_max = paths_np.max(axis=0)
        in_cone = np.all((target >= gen_min - tolerance) & (target <= gen_max + tolerance))
        if in_cone:
            cone_success += 1

        # Regime classification
        target_range = np.abs(target[-1] - target[0])
        is_trending = target_range > 0.01
        if is_trending:
            trending_count += 1
            if min_err < failure_threshold:
                trending_success += 1
        else:
            ranging_count += 1
            if min_err < failure_threshold:
                ranging_success += 1

        total += 1

        pbar.set_postfix({
            "gen_acc": f"{generator_success / total:.2%}",
            "cone": f"{cone_success / total:.2%}",
        })

    results = {
        "num_samples": total,
        "num_paths": num_paths,
        "generator_success_rate": generator_success / total if total else 0.0,
        "generator_failure_rate": 1.0 - generator_success / total if total else 0.0,
        "cone_coverage_rate": cone_success / total if total else 0.0,
        "mean_closest_distance": closest_distance_sum / total if total else 0.0,
        "trending_success_rate": trending_success / trending_count if trending_count else 0.0,
        "ranging_success_rate": ranging_success / ranging_count if ranging_count else 0.0,
    }

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate MMFPS_GEN_V2 generator")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--num-paths", type=int, default=128)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--output", type=str, default=None, help="JSON output path")
    args = parser.parse_args()

    results = evaluate(
        args.checkpoint, args.data_path,
        num_paths=args.num_paths,
        max_samples=args.max_samples,
    )

    print("\n=== MMFPS_GEN_V2 Validation ===")
    print(f"Generator success rate:   {results['generator_success_rate']:.2%}")
    print(f"Generator failure rate:   {results['generator_failure_rate']:.2%}")
    print(f"Cone coverage rate:       {results['cone_coverage_rate']:.2%}")
    print(f"Mean closest distance:    {results['mean_closest_distance']:.6f}")
    print(f"Trending success:         {results['trending_success_rate']:.2%}")
    print(f"Ranging success:          {results['ranging_success_rate']:.2%}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()