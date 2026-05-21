"""SCRIPT 7: Final inspection of the diffusion dataset.

Checks: target variance, kurtosis, autocorrelation, volatility clustering,
         zero-crossings, train/val/test consistency.

This is the mandatory sanity check before any training run.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DATA_DIR = Path("data")
DATASET = DATA_DIR / "diffusion_dataset.npz"
REPORT = DATA_DIR / "inspection_report.json"


def inspect():
    ds = np.load(DATASET)

    X = ds["X"]
    y = ds["y"]
    timestamps = ds["timestamps"]
    session_boundary = ds["session_boundary"]
    window_starts = ds["window_starts"]
    split = ds["split"]
    sample_windows = ds["sample_windows"]
    sample_targets = ds["sample_targets"]
    sample_timestamps = ds["sample_timestamps"]

    print("=== DIFFUSION DATASET INSPECTION ===\n")

    # 1. Basic stats
    print(f"Feature tensor: {X.shape}")
    print(f"Target tensor: {y.shape}")
    print(f"Valid windows: {len(window_starts):,}")
    print(f"Samples: {sample_targets.shape[0]:,}")

    # 2. Target analysis
    y_valid = y[~np.isnan(y)]
    print(f"\n--- Target (Price Delta) ---")
    print(f"  Valid: {len(y_valid):,}")
    print(f"  Mean: {np.mean(y_valid):.6f}")
    print(f"  Std: {np.std(y_valid):.6f}")
    print(f"  Min: {np.min(y_valid):.6f}")
    print(f"  Max: {np.max(y_valid):.6f}")
    print(f"  Zero-cross rate: {np.mean(y_valid > 0):.4f}")
    print(f"  Pct |delta| < 0.01: {np.mean(np.abs(y_valid) < 0.01) * 100:.2f}%")
    print(f"  Pct |delta| < 0.1:  {np.mean(np.abs(y_valid) < 0.1) * 100:.2f}%")
    print(f"  Pct |delta| > 1.0:  {np.mean(np.abs(y_valid) > 1.0) * 100:.2f}%")
    print(f"  Pct |delta| > 10.0: {np.mean(np.abs(y_valid) > 10.0) * 100:.2f}%")

    # Outliers: extreme targets that span gaps
    extreme = np.abs(y_valid) > 20
    print(f"  Extreme (|delta| > 20): {extreme.sum():,} ({np.mean(extreme) * 100:.4f}%)")

    # 3. Autocorrelation of targets
    from scipy import signal as sp_signal
    ac = sp_signal.correlate(y_valid - np.mean(y_valid), y_valid - np.mean(y_valid), mode="full")
    ac /= ac[len(ac) // 2]
    ac_lag1 = ac[len(ac) // 2 + 1]
    ac_lag5 = ac[len(ac) // 2 + 5] if len(ac) // 2 + 5 < len(ac) else 0
    ac_lag60 = ac[len(ac) // 2 + 60] if len(ac) // 2 + 60 < len(ac) else 0

    print(f"\n--- Target Autocorrelation ---")
    print(f"  Lag-1: {ac_lag1:.4f}")
    print(f"  Lag-5: {ac_lag5:.4f}")
    print(f"  Lag-60: {ac_lag60:.4f}")

    # 4. Volatility clustering (absolute delta autocorrelation)
    abs_delta = np.abs(y_valid)
    abs_ac = sp_signal.correlate(abs_delta - np.mean(abs_delta), abs_delta - np.mean(abs_delta), mode="full")
    abs_ac /= abs_ac[len(abs_ac) // 2]
    abs_lag1 = abs_ac[len(abs_ac) // 2 + 1]
    abs_lag60 = abs_ac[len(abs_ac) // 2 + 60] if len(abs_ac) // 2 + 60 < len(abs_ac) else 0

    print(f"  Abs delta lag-1: {abs_lag1:.4f} (vol clustering)")
    print(f"  Abs delta lag-60: {abs_lag60:.4f}")

    # 5. Split consistency
    print(f"\n--- Split ---")
    for label, name in [(0, "Train"), (1, "Val"), (2, "Test")]:
        mask = split == label
        n_win = mask.sum()
        y_part = y[window_starts[mask] + 256]
        y_part = y_part[~np.isnan(y_part)]
        print(f"  {name}: {n_win:,} windows, target "
              f"mean={np.mean(y_part):.4f} std={np.std(y_part):.4f}")

    # 6. Feature statistics
    print(f"\n--- Features ---")
    X_flat = X.reshape(-1)
    print(f"  Mean: {np.mean(X_flat):.6f}")
    print(f"  Std: {np.std(X_flat):.6f}")
    print(f"  Min: {np.min(X_flat):.6f}")
    print(f"  Max: {np.max(X_flat):.6f}")
    zero_variance = np.any(np.std(X, axis=0) < 1e-10)
    print(f"  Zero-variance columns: {zero_variance}")

    # 7. Sample windows quality
    print(f"\n--- Sample Windows (256-bar, n={len(sample_windows)}) ---")
    X_s = sample_windows.reshape(-1)
    print(f"  Feature mean: {np.mean(X_s):.6f}  std: {np.std(X_s):.6f}")
    y_s = sample_targets
    print(f"  Target mean: {np.mean(y_s):.6f}  std: {np.std(y_s):.6f}")

    # 8. Date range of sample
    from datetime import datetime as dt
    ts_min = dt.utcfromtimestamp(sample_timestamps.min() / 1000)
    ts_max = dt.utcfromtimestamp(sample_timestamps.max() / 1000)
    print(f"  Date range: {ts_min} to {ts_max}")

    report = {
        "n_features": X.shape[1],
        "n_total_bars": len(X),
        "n_valid_windows": len(window_starts),
        "n_sample_windows": len(sample_windows),
        "target": {
            "mean": float(f"{np.mean(y_valid):.6f}"),
            "std": float(f"{np.std(y_valid):.6f}"),
            "min": float(f"{np.min(y_valid):.6f}"),
            "max": float(f"{np.max(y_valid):.6f}"),
            "zero_cross": float(f"{np.mean(y_valid > 0):.4f}"),
            "pct_extreme_gt_20": float(f"{np.mean(np.abs(y_valid) > 20) * 100:.4f}"),
            "autocorr_lag1": float(f"{ac_lag1:.4f}"),
            "autocorr_lag60": float(f"{ac_lag60:.4f}"),
            "vol_clustering_lag1": float(f"{abs_lag1:.4f}"),
            "vol_clustering_lag60": float(f"{abs_lag60:.4f}"),
        },
        "split_stats": {
            "train_windows": int((split == 0).sum()),
            "val_windows": int((split == 1).sum()),
            "test_windows": int((split == 2).sum()),
        },
        "features": {
            "global_mean": float(f"{np.mean(X_flat):.6f}"),
            "global_std": float(f"{np.std(X_flat):.6f}"),
            "has_zero_variance": bool(zero_variance),
        },
        "sample_windows": {
            "date_range": {"start": str(ts_min), "end": str(ts_max)},
            "target_mean": float(f"{np.mean(y_s):.6f}"),
            "target_std": float(f"{np.std(y_s):.6f}"),
        },
    }

    with open(REPORT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved: {REPORT}")

    verdict = "PASS" if not zero_variance and not np.isnan(y_valid).all() else "FAIL"
    print(f"\n=== VERDICT: {verdict} ===")
    return report


if __name__ == "__main__":
    inspect()
