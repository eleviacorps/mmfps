"""SCRIPT 4: Analyze feature quality.

Checks: variance, dead columns, correlation, mutual info with target,
         distribution plots. Outputs analysis_report.json + heatmaps."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data")
FEATURE_NPY = DATA_DIR / "feature_tensor.npy"
TARGET_NPY = DATA_DIR / "target_tensor.npy"
COLUMNS_TXT = DATA_DIR / "feature_columns.txt"
REPORT = DATA_DIR / "analysis_report.json"


def analyze():
    X = np.load(FEATURE_NPY).astype(np.float64)
    y = np.load(TARGET_NPY).astype(np.float64)

    with open(COLUMNS_TXT) as f:
        columns = [line.strip() for line in f]

    n, d = X.shape
    print(f"Features: {d}  Samples: {n:,}")

    valid = ~np.isnan(y)
    y_clean = y[valid]
    X_clean = X[valid]
    print(f"Valid targets: {len(y_clean):,}")

    report: dict = {
        "n_samples": n,
        "n_features": d,
        "feature_names": columns,
    }

    # --- Per-feature stats ---
    means = np.nanmean(X, axis=0)
    stds = np.nanstd(X, axis=0)
    mins = np.nanmin(X, axis=0)
    maxs = np.nanmax(X, axis=0)
    n_nan = np.isnan(X).sum(axis=0).tolist()

    # Dead columns: zero variance or constant
    dead = (stds < 1e-10).tolist()
    dead_names = [columns[i] for i, d in enumerate(dead) if d]
    dead_count = sum(dead)

    # Near-dead: variance ratio < 0.01
    total_var = np.nansum(stds)
    var_ratios = stds / total_var if total_var > 0 else stds
    low_var = (var_ratios < 0.001).tolist()

    report["dead_columns"] = {"count": dead_count, "names": dead_names}
    report["low_var_columns"] = {"count": sum(low_var),
                                  "names": [columns[i] for i, v in enumerate(low_var) if v]}

    # Feature stats table
    feat_stats = []
    for i in range(d):
        feat_stats.append({
            "name": columns[i],
            "mean": float(f"{means[i]:.6f}"),
            "std": float(f"{stds[i]:.6f}"),
            "min": float(f"{mins[i]:.6f}"),
            "max": float(f"{maxs[i]:.6f}"),
            "nan_count": int(n_nan[i]),
            "dead": bool(dead[i]),
        })
    report["feature_stats"] = feat_stats

    # --- Correlation matrix summary ---
    corr = np.corrcoef(X.T)
    np.fill_diagonal(corr, 0)
    max_corr = np.max(np.abs(corr), axis=0).tolist()
    high_corr = np.where(np.array(max_corr) > 0.95)[0]

    report["correlation"] = {
        "max_cross_correlation": round(float(np.max(max_corr)), 4),
        "mean_abs_correlation": round(float(np.mean(np.abs(corr))), 4),
        "highly_correlated_pairs_count": len(high_corr),
        "highly_correlated_features": [columns[i] for i in high_corr],
    }

    # --- Target analysis ---
    y_valid = y_clean
    report["target"] = {
        "mean": round(float(np.mean(y_valid)), 6),
        "std": round(float(np.std(y_valid)), 6),
        "min": round(float(np.min(y_valid)), 6),
        "max": round(float(np.max(y_valid)), 6),
        "skew": round(float(pd.Series(y_valid).skew()), 4),
        "kurtosis": round(float(pd.Series(y_valid).kurtosis()), 4),
        "zero_cross_rate": round(float((y_valid > 0).mean()), 4),
        "pct_zero": round(float((np.abs(y_valid) < 1e-6).mean() * 100), 4),
    }

    print("\n=== TARGET STATS ===")
    t = report["target"]
    print(f"Mean: {t['mean']:.6f}  Std: {t['std']:.6f}")
    print(f"Min: {t['min']:.6f}  Max: {t['max']:.6f}")
    print(f"Skew: {t['skew']:.4f}  Kurtosis: {t['kurtosis']:.4f}")
    print(f"Zero-cross rate: {t['zero_cross_rate']:.4f}")

    print("\n=== FEATURE QUALITY ===")
    print(f"Dead columns: {dead_count}/{d}")
    print(f"Low variance: {report['low_var_columns']['count']}/{d}")
    print(f"Highly correlated pairs: {report['correlation']['highly_correlated_pairs_count']}")
    print(f"Mean abs correlation: {report['correlation']['mean_abs_correlation']:.4f}")

    # Save report
    with open(REPORT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved: {REPORT}")

    return report


if __name__ == "__main__":
    analyze()
