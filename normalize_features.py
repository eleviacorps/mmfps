"""SCRIPT 5: Normalize features safely.

Strategy: Robust scaling (median/IQR) for each feature.
- No z-score on raw prices (we use log returns instead)
- No global centering of price levels
- Features normalized independently, preserving relationships
- Outliers clipped at 5 sigma after robust scaling

Output: normalized_feature_tensor.npy + scaler_metadata.json"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DATA_DIR = Path("data")
FEATURE_NPY = DATA_DIR / "feature_tensor.npy"
COLUMNS_TXT = DATA_DIR / "feature_columns.txt"
OUTPUT_NPY = DATA_DIR / "normalized_feature_tensor.npy"
SCALER_META = DATA_DIR / "scaler_metadata.json"


def normalize():
    X = np.load(FEATURE_NPY).astype(np.float64)
    with open(COLUMNS_TXT) as f:
        columns = [line.strip() for line in f]

    n, d = X.shape
    print(f"Normalizing {n:,} samples x {d} features")

    X_norm = np.empty_like(X)
    scaler_info = {}

    for i in range(d):
        col = X[:, i]
        valid = col[~np.isnan(col)]

        # Robust scaling: (x - median) / IQR
        median = float(np.median(valid))
        q1 = float(np.percentile(valid, 25))
        q3 = float(np.percentile(valid, 75))
        iqr = q3 - q1

        if iqr > 1e-10:
            scaled = (col - median) / iqr
        else:
            scaled = col - median  # fallback: just center

        # Clip extreme outliers at 5 sigma
        std = float(np.std(valid))
        clip = 5.0 * std / (iqr if iqr > 1e-10 else 1.0)
        scaled = np.clip(scaled, -clip, clip)

        # Store metadata
        scaler_info[columns[i]] = {
            "median": round(median, 8),
            "q1": round(q1, 8),
            "q3": round(q3, 8),
            "iqr": round(iqr if iqr > 0 else 1.0, 8),
            "clip_threshold": round(clip, 4),
            "mean_after_norm": round(float(np.mean(scaled)), 6),
            "std_after_norm": round(float(np.std(scaled)), 6),
        }

        X_norm[:, i] = scaled

    # Final statistics
    final_mean = float(np.mean(X_norm))
    final_std = float(np.std(X_norm))
    clip_count = int(np.sum(np.abs(X_norm) >= clip * 0.999))
    clip_pct = round(clip_count / (n * d) * 100, 4)

    meta = {
        "n_samples": n,
        "n_features": d,
        "method": "robust_scaling_median_iqr",
        "clip_std": clip,
        "final_mean": round(final_mean, 6),
        "final_std": round(final_std, 6),
        "clipped_values": clip_count,
        "clipped_pct": clip_pct,
        "per_feature": scaler_info,
    }

    print(f"\nFinal mean: {final_mean:.6f}  std: {final_std:.6f}")
    print(f"Clipped values: {clip_count:,} ({clip_pct}%)")

    np.save(OUTPUT_NPY, X_norm.astype(np.float32))
    with open(SCALER_META, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved: {OUTPUT_NPY}")
    print(f"Saved: {SCALER_META}")

    return X_norm


if __name__ == "__main__":
    normalize()
