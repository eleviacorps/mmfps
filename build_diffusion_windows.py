"""SCRIPT 6: Pack temporal windows for diffusion training.

Window sizes: short=64, mid=256, long=1024 bars
Target: forward price delta (already computed)
Strict temporal order, no leakage across session boundaries.
Train/val/test split: 80/10/10 temporal.

Output: diffusion_dataset.npz with keys:
  - X_short, X_mid, X_long: (N, window, features)
  - y: (N,) target deltas
  - timestamps: (N,) original timestamps
  - split: (N,) 0=train, 1=val, 2=test
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DATA_DIR = Path("data")
FEATURE_NPY = DATA_DIR / "normalized_feature_tensor.npy"
TARGET_NPY = DATA_DIR / "target_tensor.npy"
TIMESTAMPS_NPY = DATA_DIR / "feature_timestamps.npy"
SESSION_NPY = DATA_DIR / "session_boundary.npy"
OUTPUT = DATA_DIR / "diffusion_dataset.npz"
REPORT = DATA_DIR / "dataset_report.json"

WINDOWS = {"short": 64, "mid": 256, "long": 1024}

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10


def pack_windows():
    X = np.load(FEATURE_NPY).astype(np.float32)
    y = np.load(TARGET_NPY).astype(np.float32)
    timestamps = np.load(TIMESTAMPS_NPY)
    session_boundary = np.load(SESSION_NPY).astype(bool)

    n = len(X)
    n_features = X.shape[1]

    # Strategy: store flat tensors + window metadata for on-the-fly extraction
    # This avoids massive memory blowup from sliding windows

    # Compute valid window start indices within each session
    print("Finding valid window positions...")
    valid_starts = []
    i = 0
    while i < n:
        while i < n and session_boundary[i]:
            i += 1
        if i >= n:
            break
        j = i
        while j < n and not session_boundary[j]:
            j += 1
        session_end = j
        for k in range(i, session_end - 1024):  # enough room for longest window + target
            valid_starts.append(k)
        i = session_end

    start_idx = np.array(valid_starts, dtype=np.int32)
    n_windows = len(start_idx)
    print(f"Valid window starts: {n_windows:,}")

    # Temporal split
    n_train = int(n_windows * TRAIN_RATIO)
    n_val = int(n_windows * VAL_RATIO)
    split_labels = np.full(n_windows, 2, dtype=np.int32)
    split_labels[:n_train] = 0
    split_labels[n_train:n_train + n_val] = 1
    print(f"Temporal split: train={n_train:,} val={n_val:,} test={n_windows - n_train - n_val:,}")

    # Sample a small subset for validation/inspection (10K windows)
    rng = np.random.default_rng(42)
    val_start_idx = start_idx[split_labels == 1]
    sample_n = min(10000, len(val_start_idx))
    sample_idx = rng.choice(len(val_start_idx), sample_n, replace=False)
    sample_starts = val_start_idx[sample_idx]

    print(f"Building {sample_n:,} sample mid-windows (256) for inspection...")
    win_size = 256
    idx_grid = np.arange(win_size)[None, :] + sample_starts[:, None]
    sample_windows = X[idx_grid.astype(np.int32)].astype(np.float32)
    sample_targets = y[sample_starts + win_size].astype(np.float32)
    sample_timestamps = timestamps[sample_starts + win_size]
    print(f"  Sample windows: {sample_windows.shape}")
    print(f"  Sample targets: {sample_targets.shape}")
    # Recompute: target for window ending at position k is y[k]
    # We already stored it in tgt_arr during the loop
    # Target stats
    non_nan = sample_targets[~np.isnan(sample_targets)]
    print(f"\nTarget stats (sample):")
    print(f"  Valid: {len(non_nan):,}")
    print(f"  Mean: {np.mean(non_nan):.6f}  Std: {np.std(non_nan):.6f}")
    print(f"  Min: {np.min(non_nan):.6f}  Max: {np.max(non_nan):.6f}")

    # Save dataset: flat tensors + window start indices (on-the-fly extraction at training time)
    np.savez(OUTPUT,
             X=X, y=y, timestamps=timestamps,
             session_boundary=session_boundary,
             window_starts=start_idx, split=split_labels,
             sample_windows=sample_windows,
             sample_targets=sample_targets,
             sample_timestamps=sample_timestamps)
    print(f"\nSaved: {OUTPUT}")

    loaded = np.load(OUTPUT)
    for key in loaded:
        arr = loaded[key]
        print(f"  {key}: {arr.shape}, {arr.dtype}, "
              f"{arr.nbytes / 1e6:.1f}MB")

    window_sizes_str = ", ".join(f"{k}={v}" for k, v in WINDOWS.items())
    report = {
        "total_valid_windows": n_windows,
        "window_sizes": WINDOWS,
        "n_features": n_features,
        "strategy": "on_the_fly_extraction",
        "train_samples": int(n_train),
        "val_samples": int(n_val),
        "test_samples": int(n_windows - n_train - n_val),
        "sample_windows_saved": sample_n,
        "target_sample": {
            "mean": float(np.nanmean(sample_targets)),
            "std": float(np.nanstd(sample_targets)),
            "min": float(np.nanmin(sample_targets)),
            "max": float(np.nanmax(sample_targets)),
            "nan_count": int(np.isnan(sample_targets).sum()),
        },
        "split_ratios": {
            "train": TRAIN_RATIO,
            "val": VAL_RATIO,
            "test": round(1 - TRAIN_RATIO - VAL_RATIO, 4),
        },
    }

    with open(REPORT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report: {REPORT}")

    return report


if __name__ == "__main__":
    pack_windows()
