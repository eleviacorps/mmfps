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

from config import BehaviorGenConfig
from dataset import PathDataset
from generator import BehaviorDiffusionGenerator


@torch.no_grad()
def evaluate(
    checkpoint_path: str,
    config: Optional[BehaviorGenConfig] = None,
    num_paths: int = 128,
    max_samples: int = 1000,
    device: Optional[str] = None,
    failure_threshold: float = 0.01,
    tolerance: float = 0.02,
) -> dict:
    """Run validation evaluation."""
    cfg = config or BehaviorGenConfig()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)

    ckpt = torch.load(checkpoint_path, map_location=device_t, weights_only=False)
    model = BehaviorDiffusionGenerator(ckpt.get("config", cfg)).to(device_t)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.scheduler.noise_scale_val = 1.0

    val_ds = PathDataset(model.config, split="val")
    n_samples = min(max_samples, len(val_ds))

    # Metrics accumulators
    generator_success = 0     # >=1 path with MSE < threshold
    calibrated_success = 0
    cone_success = 0          # real inside generated min/max range
    tight_cone_success = 0
    closest_distance_sum = 0.0
    scaled_distance_sum = 0.0
    scaled_distances = []
    variance_ratios = []
    worst_samples = []
    direction_hits = 0
    best_direction_hits = 0
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
        target = sample.target.numpy() / model.config.target_scale

        paths, _, _ = model.generate(short, mid, long, num_paths=num_paths)
        paths_np = paths[0].cpu().numpy()  # (num_paths, T)

        # Squared error for each path
        sq_errors = ((paths_np - target) ** 2).mean(axis=-1)  # (num_paths,)
        direction_match = np.sign(paths_np[:, -1] - paths_np[:, 0]) == np.sign(target[-1] - target[0])
        magnitude_errors = np.abs(paths_np[:, -1] - target[-1])
        vol_errors = np.abs(paths_np.std(axis=-1) - target.std())
        structure_errors = np.mean(
            (np.cumsum(paths_np, axis=-1) - np.cumsum(target)[None, :]) ** 2,
            axis=-1,
        )
        min_err = sq_errors.min()
        target_var = float(np.var(target) + 1e-12)
        scaled_min_err = float(min_err / target_var)
        gen_var = float(np.var(paths_np))
        variance_ratio = float(gen_var / target_var)
        composite = (
            sq_errors
            + 0.25 * magnitude_errors
            + 0.25 * vol_errors
            + 0.25 * structure_errors
            + 0.05 * (~direction_match)
        )
        best_idx = composite.argmin()
        mse_best_idx = int(sq_errors.argmin())

        closest_distance_sum += float(min_err)
        scaled_distance_sum += scaled_min_err
        scaled_distances.append(scaled_min_err)
        variance_ratios.append(variance_ratio)

        # Generator success: at least one path is close
        if min_err < failure_threshold:
            generator_success += 1
        if scaled_min_err < 1.0:
            calibrated_success += 1

        # Cone coverage: real inside generated envelope
        gen_min = paths_np.min(axis=0)
        gen_max = paths_np.max(axis=0)
        in_cone = np.all((target >= gen_min - tolerance) & (target <= gen_max + tolerance))
        if in_cone:
            cone_success += 1
        tight_tol = 0.25 * float(np.std(target) + 1e-12)
        tight_in_cone = np.all((target >= gen_min - tight_tol) & (target <= gen_max + tight_tol))
        if tight_in_cone:
            tight_cone_success += 1

        target_direction = np.sign(target[-1] - target[0])
        path_directions = np.sign(paths_np[:, -1] - paths_np[:, 0])
        worst_samples.append({
            "sample_index": int(i),
            "scaled_min_error": scaled_min_err,
            "min_error": float(min_err),
            "target_var": target_var,
            "target_std": float(np.std(target)),
            "generated_var": gen_var,
            "variance_ratio": variance_ratio,
            "target_abs_max": float(np.max(np.abs(target))),
            "generated_abs_max": float(np.max(np.abs(paths_np))),
            "best_mse_direction_match": bool(path_directions[mse_best_idx] == target_direction),
            "mse_best_idx": mse_best_idx,
        })
        if np.any(path_directions == target_direction):
            direction_hits += 1
        if path_directions[mse_best_idx] == target_direction:
            best_direction_hits += 1

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

    scaled_arr = np.asarray(scaled_distances, dtype=np.float64)
    variance_arr = np.asarray(variance_ratios, dtype=np.float64)
    worst_by_scaled = sorted(worst_samples, key=lambda x: x["scaled_min_error"], reverse=True)[:20]
    worst_by_variance = sorted(worst_samples, key=lambda x: x["variance_ratio"], reverse=True)[:20]

    results = {
        "num_samples": total,
        "num_paths": num_paths,
        "generator_success_rate": generator_success / total if total else 0.0,
        "calibrated_success_rate_scaled_mse_lt_1": calibrated_success / total if total else 0.0,
        "generator_failure_rate": 1.0 - generator_success / total if total else 0.0,
        "cone_coverage_rate": cone_success / total if total else 0.0,
        "tight_cone_coverage_rate": tight_cone_success / total if total else 0.0,
        "mean_closest_distance": closest_distance_sum / total if total else 0.0,
        "mean_scaled_closest_distance": scaled_distance_sum / total if total else 0.0,
        "mean_variance_ratio": float(np.mean(variance_ratios)) if variance_ratios else 0.0,
        "median_variance_ratio": float(np.median(variance_ratios)) if variance_ratios else 0.0,
        "scaled_closest_distance_percentiles": {
            "p50": float(np.percentile(scaled_arr, 50)) if total else 0.0,
            "p75": float(np.percentile(scaled_arr, 75)) if total else 0.0,
            "p90": float(np.percentile(scaled_arr, 90)) if total else 0.0,
            "p95": float(np.percentile(scaled_arr, 95)) if total else 0.0,
            "p99": float(np.percentile(scaled_arr, 99)) if total else 0.0,
            "max": float(np.max(scaled_arr)) if total else 0.0,
        },
        "variance_ratio_percentiles": {
            "p50": float(np.percentile(variance_arr, 50)) if total else 0.0,
            "p75": float(np.percentile(variance_arr, 75)) if total else 0.0,
            "p90": float(np.percentile(variance_arr, 90)) if total else 0.0,
            "p95": float(np.percentile(variance_arr, 95)) if total else 0.0,
            "p99": float(np.percentile(variance_arr, 99)) if total else 0.0,
            "max": float(np.max(variance_arr)) if total else 0.0,
        },
        "outlier_rates": {
            "scaled_mse_gt_10": float(np.mean(scaled_arr > 10.0)) if total else 0.0,
            "scaled_mse_gt_100": float(np.mean(scaled_arr > 100.0)) if total else 0.0,
            "scaled_mse_gt_1000": float(np.mean(scaled_arr > 1000.0)) if total else 0.0,
            "variance_ratio_gt_10": float(np.mean(variance_arr > 10.0)) if total else 0.0,
            "variance_ratio_gt_100": float(np.mean(variance_arr > 100.0)) if total else 0.0,
            "variance_ratio_gt_1000": float(np.mean(variance_arr > 1000.0)) if total else 0.0,
        },
        "worst_by_scaled_error": worst_by_scaled,
        "worst_by_variance_ratio": worst_by_variance,
        "direction_coverage_rate": direction_hits / total if total else 0.0,
        "best_mse_direction_match_rate": best_direction_hits / total if total else 0.0,
        "trending_success_rate": trending_success / trending_count if trending_count else 0.0,
        "ranging_success_rate": ranging_success / ranging_count if ranging_count else 0.0,
    }

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate MMFPS_GEN_V2 generator")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num-paths", type=int, default=128)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--output", type=str, default=None, help="JSON output path")
    args = parser.parse_args()

    results = evaluate(
        args.checkpoint,
        num_paths=args.num_paths,
        max_samples=args.max_samples,
    )

    print("\n=== MMFPS_GEN_V2 Validation ===")
    print(f"Generator success rate:   {results['generator_success_rate']:.2%}")
    print(f"Generator failure rate:   {results['generator_failure_rate']:.2%}")
    print(f"Cone coverage rate:       {results['cone_coverage_rate']:.2%}")
    print(f"Tight cone coverage:      {results['tight_cone_coverage_rate']:.2%}")
    print(f"Mean closest distance:    {results['mean_closest_distance']:.6f}")
    print(f"Mean scaled closest dist: {results['mean_scaled_closest_distance']:.4f}")
    print(f"Calibrated success <1x:   {results['calibrated_success_rate_scaled_mse_lt_1']:.2%}")
    print(f"Variance ratio mean/med:  {results['mean_variance_ratio']:.3f} / {results['median_variance_ratio']:.3f}")
    print(
        "Scaled dist p50/p95/p99: "
        f"{results['scaled_closest_distance_percentiles']['p50']:.3f} / "
        f"{results['scaled_closest_distance_percentiles']['p95']:.3f} / "
        f"{results['scaled_closest_distance_percentiles']['p99']:.3f}"
    )
    print(
        "Variance p50/p95/p99:    "
        f"{results['variance_ratio_percentiles']['p50']:.3f} / "
        f"{results['variance_ratio_percentiles']['p95']:.3f} / "
        f"{results['variance_ratio_percentiles']['p99']:.3f}"
    )
    print(f"Direction coverage:       {results['direction_coverage_rate']:.2%}")
    print(f"Best-MSE direction match: {results['best_mse_direction_match_rate']:.2%}")
    print(f"Trending success:         {results['trending_success_rate']:.2%}")
    print(f"Ranging success:          {results['ranging_success_rate']:.2%}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
