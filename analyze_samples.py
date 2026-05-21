"""DDIM sample quality analysis — fixed metrics and data comparison."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from config import BehaviorGenConfig
from dataset import PathDataset
from generator import BehaviorDiffusionGenerator


@torch.no_grad()
def analyze_samples(
    checkpoint_path: str,
    output_path: str,
    num_paths: int = 128,
    num_samples: int = 100,
    device: str = "cuda",
):
    device_t = torch.device(device)

    ckpt = torch.load(checkpoint_path, map_location=device_t, weights_only=False)
    cfg = ckpt.get("config", BehaviorGenConfig())
    model = BehaviorDiffusionGenerator(cfg).to(device_t)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.scheduler.noise_scale_val = 1.0

    val_ds = PathDataset(cfg, split="val")
    n = min(num_samples, len(val_ds))

    all_targets = []
    all_paths = []

    for i in range(n):
        sample = val_ds[i]
        short = sample.short_seq.unsqueeze(0).to(device_t)
        mid = sample.mid_seq.unsqueeze(0).to(device_t)
        long = sample.long_seq.unsqueeze(0).to(device_t)
        all_targets.append(sample.target.numpy())

        paths, _, _ = model.generate(short, mid, long, num_paths=num_paths)
        all_paths.append(paths[0].cpu().numpy())

    all_targets = np.stack(all_targets)
    all_paths = np.stack(all_paths)

    results = {}

    gen_all = all_paths.reshape(-1)
    real_all = all_targets.reshape(-1)
    p01, p99 = np.percentile(real_all, [1, 99])
    real_finite = real_all[np.isfinite(real_all)]
    results["real_return_mean"] = float(np.mean(real_finite))
    results["real_return_std"] = float(np.std(real_finite))
    results["real_return_p01"] = float(np.percentile(real_finite, 1))
    results["real_return_p99"] = float(np.percentile(real_finite, 99))

    gen_finite = gen_all[np.isfinite(gen_all)]
    results["gen_return_mean"] = float(np.mean(gen_finite))
    results["gen_return_std"] = float(np.std(gen_finite))
    results["gen_return_p01"] = float(np.percentile(gen_finite, 1))
    results["gen_return_p99"] = float(np.percentile(gen_finite, 99))

    ks_stat, ks_p = stats.ks_2samp(gen_finite, real_finite)
    results["ks_statistic"] = float(ks_stat)
    results["ks_pvalue"] = float(ks_p)

    cross_path_var = all_paths.var(axis=1).mean(axis=-1)
    results["mean_cross_path_variance"] = float(cross_path_var.mean())
    results["total_gen_variance"] = float(gen_finite.var())
    results["total_real_variance"] = float(real_finite.var())
    results["variance_ratio"] = float(gen_finite.var() / (real_finite.var() + 1e-8))

    gen_d2 = np.diff(all_paths, n=2, axis=-1)
    real_d2 = np.diff(all_targets, n=2, axis=-1)
    results["gen_second_diff_mean"] = float(np.abs(gen_d2).mean())
    results["real_second_diff_mean"] = float(np.abs(real_d2).mean())

    def path_autocorr(paths, lag=1):
        T = paths.shape[-1]
        if T <= lag:
            return 0.0
        flat = paths.reshape(-1, T)
        ac = np.array([
            np.corrcoef(flat[j, :-lag], flat[j, lag:])[0, 1]
            for j in range(flat.shape[0])
            if np.isfinite(flat[j]).all()
        ])
        ac = ac[np.isfinite(ac)]
        return float(np.mean(ac)) if len(ac) > 0 else 0.0

    results["gen_autocorr_lag1"] = path_autocorr(all_paths, 1)
    results["gen_autocorr_lag2"] = path_autocorr(all_paths, 2)
    results["real_autocorr_lag1"] = path_autocorr(all_targets[:, None, :], 1)
    results["real_autocorr_lag2"] = path_autocorr(all_targets[:, None, :], 2)

    gen_abs_ret = np.abs(np.diff(all_paths, axis=-1))
    real_abs_ret = np.abs(np.diff(all_targets, axis=-1))

    def vol_clustering(abs_ret, lag=1):
        flat = abs_ret.reshape(-1, abs_ret.shape[-1])
        ac = np.array([
            np.corrcoef(flat[j, :-lag], flat[j, lag:])[0, 1]
            for j in range(flat.shape[0])
            if np.isfinite(flat[j]).all()
        ])
        ac = ac[np.isfinite(ac)]
        return float(np.mean(ac)) if len(ac) > 0 else 0.0

    results["gen_vol_clustering_lag1"] = vol_clustering(gen_abs_ret, 1)
    results["real_vol_clustering_lag1"] = vol_clustering(real_abs_ret, 1)

    results["gen_kurtosis"] = float(stats.kurtosis(gen_finite, fisher=False))
    results["real_kurtosis"] = float(stats.kurtosis(real_finite, fisher=False))

    sq_errors = ((all_paths - all_targets[:, None, :]) ** 2).sum(axis=-1)
    best_errors = sq_errors.min(axis=-1)
    results["mean_best_path_mse"] = float(best_errors.mean())
    real_var_per_sample = np.array([np.var(t) for t in all_targets])
    scaled_errors = best_errors / (real_var_per_sample + 1e-8)
    results["mean_scaled_best_mse"] = float(scaled_errors.mean())

    print("\n=== Sample Quality Analysis ===")
    print(f"Distribution:")
    print(f"  Gen return:  mean={results['gen_return_mean']:.4f}, std={results['gen_return_std']:.4f}")
    print(f"  Real return: mean={results['real_return_mean']:.4f}, std={results['real_return_std']:.4f}")
    print(f"  Gen p01/p99: {results['gen_return_p01']:.4f} / {results['gen_return_p99']:.4f}")
    print(f"  Real p01/p99:{results['real_return_p01']:.4f} / {results['real_return_p99']:.4f}")
    print(f"  KS stat:     {results['ks_statistic']:.4f} (p={results['ks_pvalue']:.4f})")
    print(f"  Variance ratio (gen/real): {results['variance_ratio']:.3f}")
    print(f"Diversity:")
    print(f"  Cross-path variance: {results['mean_cross_path_variance']:.4f}")
    print(f"Temporal:")
    print(f"  Gen autocorr lag1:  {results['gen_autocorr_lag1']:.4f}")
    print(f"  Real autocorr lag1: {results['real_autocorr_lag1']:.4f}")
    print(f"  Gen 2nd diff:  {results['gen_second_diff_mean']:.6f}")
    print(f"  Real 2nd diff: {results['real_second_diff_mean']:.6f}")
    print(f"Volatility:")
    print(f"  Gen vol cluster lag1:  {results['gen_vol_clustering_lag1']:.4f}")
    print(f"  Real vol cluster lag1: {results['real_vol_clustering_lag1']:.4f}")
    print(f"Tails:")
    print(f"  Gen kurtosis:  {results['gen_kurtosis']:.2f}")
    print(f"  Real kurtosis: {results['real_kurtosis']:.2f}")
    print(f"Generator:")
    print(f"  Best path MSE: {results['mean_best_path_mse']:.4f}")
    print(f"  Scaled best MSE: {results['mean_scaled_best_mse']:.4f}")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="sample_analysis.json")
    parser.add_argument("--num-paths", type=int, default=128)
    parser.add_argument("--num-samples", type=int, default=100)
    args = parser.parse_args()
    analyze_samples(
        args.checkpoint, args.output,
        num_paths=args.num_paths, num_samples=args.num_samples,
    )
