"""SCRIPT 3: Build features + targets from clean XAUUSD M1 data.

Delegates to the `features` package for all computation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from features.orchestrator import build_all_features

DATA_DIR = Path("data")
CLEAN_CSV = DATA_DIR / "xauusd_m1_clean.csv"
FEATURE_NPY = DATA_DIR / "feature_tensor.npy"
TARGET_NPY = DATA_DIR / "target_tensor.npy"
TARGET_VALID_NPY = DATA_DIR / "target_valid.npy"
SESSION_ID_NPY = DATA_DIR / "session_id.npy"
TIMESTAMPS_NPY = DATA_DIR / "feature_timestamps.npy"
FEATURE_COLUMNS = DATA_DIR / "feature_columns.txt"


def main():
    df = pd.read_csv(CLEAN_CSV)
    print(f"Loaded {len(df):,} bars")

    features, target, target_valid, stats = build_all_features(df)

    # Remove rows with NaN features (initial warm-up period)
    valid = features.dropna().index
    feature_tensor = features.loc[valid].values.astype(np.float32)
    target_tensor = target[valid].astype(np.float32)
    target_valid_arr = target_valid[valid].astype(np.int8)
    session_ids = df["session_id"].values[valid].astype(np.int32)
    timestamps = df["timestamp"].values[valid].astype(np.int64)

    print(f"\nFeature tensor: {feature_tensor.shape}")
    print(f"Target tensor: {target_tensor.shape}")
    print(f"Features: {features.shape[1]}")

    np.save(FEATURE_NPY, feature_tensor)
    np.save(TARGET_NPY, target_tensor)
    np.save(TARGET_VALID_NPY, target_valid_arr)
    np.save(SESSION_ID_NPY, session_ids)
    np.save(TIMESTAMPS_NPY, timestamps)

    with open(FEATURE_COLUMNS, "w") as f:
        for col in features.columns:
            f.write(col + "\n")

    print(f"\nSaved all tensors to {DATA_DIR}")

    # Summary
    yv = target_tensor[target_valid_arr == 1]
    print(f"\n--- Target Summary (valid, N={len(yv):,}) ---")
    print(f"Mean: {np.mean(yv):.8f}  Std: {np.std(yv):.8f}")
    print(f"Min: {np.min(yv):.8f}  Max: {np.max(yv):.8f}")
    print(f"Skew: {pd.Series(yv).skew():.2f}  Kurt: {pd.Series(yv).kurtosis():.2f}")
    print(f"Zero-cross: {np.mean(yv > 0):.4f}")
    print(f"Feature columns: {stats['n_features']}")
    print(f"Target clipped: {stats['target']['clipped_pct']:.4f}%")


if __name__ == "__main__":
    main()
