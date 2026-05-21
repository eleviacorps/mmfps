"""SCRIPT 5: Normalize features safely.

Strategy: Robust scaling (median/IQR) for each feature.
- No z-score on raw prices (we use log returns instead)
- No global centering of price levels
- Features normalized independently, preserving relationships
- Outliers clipped at 5 sigma after robust scaling
- Scaler is fit on the training period only to avoid validation/test leakage

Output: normalized_feature_tensor.npy + scaler_metadata.json"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DATA_DIR = Path("data")
FEATURE_NPY = DATA_DIR / "feature_tensor.npy"
TIMESTAMPS_NPY = DATA_DIR / "feature_timestamps.npy"
COLUMNS_TXT = DATA_DIR / "feature_columns.txt"
OUTPUT_NPY = DATA_DIR / "normalized_feature_tensor.npy"
SCALER_META = DATA_DIR / "scaler_metadata.json"
VAL_SPLIT_TS_MS = 1672531200 * 1000


def normalize():
    X = np.load(FEATURE_NPY, mmap_mode="r")
    timestamps = np.load(TIMESTAMPS_NPY, mmap_mode="r")
    with open(COLUMNS_TXT) as f:
        columns = [line.strip() for line in f]

    n, d = X.shape
    print(f"Normalizing {n:,} samples x {d} features")
    train_mask = timestamps < VAL_SPLIT_TS_MS
    print(f"Fitting robust scalers on {int(train_mask.sum()):,} training-period rows")

    X_norm = np.lib.format.open_memmap(OUTPUT_NPY, dtype=np.float32, mode="w+", shape=(n, d))
    scaler_info = {}
    total_clipped = 0

    for i in range(d):
        col_train = np.asarray(X[train_mask, i], dtype=np.float64)
        valid = col_train[~np.isnan(col_train)]

        # Robust scaling: (x - median) / IQR
        median = float(np.median(valid))
        q1 = float(np.percentile(valid, 25))
        q3 = float(np.percentile(valid, 75))
        iqr = q3 - q1

        if iqr > 1e-10:
            scale = iqr
        else:
            scale = 1.0

        # Clip extreme outliers at 5 train-period standard deviations.
        std = float(np.std(valid))
        clip = max(5.0 * std / scale, 1.0)

        chunk = 500_000
        clipped_count = 0
        col_sum = 0.0
        col_sq_sum = 0.0
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            scaled = (np.asarray(X[start:end, i], dtype=np.float64) - median) / scale
            scaled = np.nan_to_num(scaled, nan=0.0, posinf=clip, neginf=-clip)
            clipped = np.clip(scaled, -clip, clip)
            clipped_count += int(np.sum(clipped != scaled))
            X_norm[start:end, i] = clipped.astype(np.float32)
            col_sum += float(clipped.sum())
            col_sq_sum += float(np.square(clipped).sum())

        col_mean = col_sum / n
        col_std = max(col_sq_sum / n - col_mean * col_mean, 0.0) ** 0.5
        total_clipped += clipped_count

        # Store metadata
        scaler_info[columns[i]] = {
            "median": round(median, 8),
            "q1": round(q1, 8),
            "q3": round(q3, 8),
            "iqr": round(scale, 8),
            "clip_threshold": round(clip, 4),
            "mean_after_norm": round(col_mean, 6),
            "std_after_norm": round(col_std, 6),
            "clipped_values": clipped_count,
        }

    # Final statistics
    final_mean = float(np.mean(X_norm))
    final_std = float(np.std(X_norm))
    clip_pct = round(total_clipped / (n * d) * 100, 4)

    meta = {
        "n_samples": n,
        "n_features": d,
        "method": "robust_scaling_median_iqr",
        "fit_period": "train_only_before_2023_01_01_utc",
        "val_split_ts_ms": VAL_SPLIT_TS_MS,
        "final_mean": round(final_mean, 6),
        "final_std": round(final_std, 6),
        "clipped_values": total_clipped,
        "clipped_pct": clip_pct,
        "per_feature": scaler_info,
    }

    print(f"\nFinal mean: {final_mean:.6f}  std: {final_std:.6f}")
    print(f"Clipped values: {total_clipped:,} ({clip_pct}%)")

    X_norm.flush()
    with open(SCALER_META, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved: {OUTPUT_NPY}")
    print(f"Saved: {SCALER_META}")

    return X_norm


if __name__ == "__main__":
    normalize()
