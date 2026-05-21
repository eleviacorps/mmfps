"""SCRIPT 6: Pack temporal windows for diffusion training.

Strategy:
- Flat feature tensor + session IDs → sliding windows with deterministic indexing
- Each window entirely within a single session (no cross-session mixing)
- Target = log return at last position within window
- Features saved as separate .npy (mmap-compatible); metadata as .npz

Output:
  diffusion_features.npy  — (n_windows, W, F) float32
  diffusion_metadata.npz  — targets, target_valid, timestamps, session_ids, split
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime

import numpy as np

DATA_DIR = Path("data")
FEATURE_NPY = DATA_DIR / "feature_tensor.npy"
TARGET_NPY = DATA_DIR / "target_tensor.npy"
TARGET_VALID_NPY = DATA_DIR / "target_valid.npy"
SESSION_ID_NPY = DATA_DIR / "session_id.npy"
TIMESTAMPS_NPY = DATA_DIR / "feature_timestamps.npy"
OUTPUT_FEATURES = DATA_DIR / "diffusion_features.npy"
OUTPUT_METADATA = DATA_DIR / "diffusion_metadata.npz"
TEMP_MEMMAP = DATA_DIR / "_diffusion_memmap.npy"

WINDOW_LENGTH = 128
WINDOW_STRIDE = 16


def generate_windows():
    features = np.load(FEATURE_NPY)
    targets = np.load(TARGET_NPY)
    target_valid = np.load(TARGET_VALID_NPY)
    session_ids = np.load(SESSION_ID_NPY)
    timestamps = np.load(TIMESTAMPS_NPY)

    N, F = features.shape
    W = WINDOW_LENGTH

    print(f"Features: {N:,} × {F}")
    print(f"Window length: {W}, stride: {WINDOW_STRIDE}")

    # Build start indices per session
    unique_sessions = np.unique(session_ids)
    start_indices = []

    for sid in unique_sessions:
        mask = session_ids == sid
        session_start = int(np.where(mask)[0][0])
        session_end = int(np.where(mask)[0][-1])
        last_start = session_end - W + 1
        if last_start < session_start:
            continue
        starts = np.arange(session_start, last_start + 1, WINDOW_STRIDE)
        start_indices.append(starts)

    all_starts = np.concatenate(start_indices)
    n_windows = len(all_starts)
    print(f"Total windows: {n_windows:,}")
    last_indices = all_starts + W - 1

    # Memory-mapped features file (directly to final path)
    mmap = np.memmap(TEMP_MEMMAP, dtype=np.float32, mode="w+",
                     shape=(n_windows, W, F))
    target_vals = np.empty(n_windows, dtype=np.float32)
    target_valid_flags = np.empty(n_windows, dtype=np.int8)
    window_tstamps = np.empty(n_windows, dtype=np.int64)
    window_sids = np.empty(n_windows, dtype=np.int32)

    chunk = 10000
    for start_idx in range(0, n_windows, chunk):
        end_idx = min(start_idx + chunk, n_windows)
        batch_starts = all_starts[start_idx:end_idx]
        batch_last = last_indices[start_idx:end_idx]
        for i, (s, l) in enumerate(zip(batch_starts, batch_last)):
            local = start_idx + i
            mmap[local] = features[s:s + W]
            target_vals[local] = targets[l]
            target_valid_flags[local] = target_valid[l]
            window_tstamps[local] = timestamps[l]
            window_sids[local] = session_ids[l]
        if (start_idx // chunk) % 5 == 0:
            print(f"  Written {end_idx:,}/{n_windows:,}")

    mmap.flush()
    del mmap

    # Rewrite as proper npy with header (must load into memory once)
    print("  Writing proper .npy format (loading from memmap)...")
    feat_data = np.array(np.memmap(TEMP_MEMMAP, dtype=np.float32, mode="r",
                                   shape=(n_windows, W, F)))
    np.save(OUTPUT_FEATURES, feat_data)
    TEMP_MEMMAP.unlink(missing_ok=True)
    print(f"Features saved: {OUTPUT_FEATURES}")

    # Split
    split = np.full(n_windows, -1, dtype=np.int8)
    val_cutoff = int(datetime(2023, 1, 1).timestamp() * 1000)
    test_cutoff = int(datetime(2025, 1, 1).timestamp() * 1000)
    split[window_tstamps < val_cutoff] = 0
    split[(window_tstamps >= val_cutoff) & (window_tstamps < test_cutoff)] = 1
    split[window_tstamps >= test_cutoff] = 2
    print(f"Split: train={(split==0).sum():,}  val={(split==1).sum():,}  test={(split==2).sum():,}")

    # Metadata
    np.savez_compressed(
        OUTPUT_METADATA,
        targets=target_vals,
        target_valid=target_valid_flags,
        timestamps=window_tstamps,
        session_ids=window_sids,
        split=split,
        n_windows=n_windows,
        window_length=W,
        stride=WINDOW_STRIDE,
        total_bars=N,
        n_features=F,
    )
    print(f"Metadata saved: {OUTPUT_METADATA}")
    meta_size = OUTPUT_METADATA.stat().st_size
    print(f"Metadata compressed size: {meta_size / 1e6:.2f} MB")

    # Stats
    yv = target_vals[target_valid_flags == 1]
    print(f"\n--- Target stats (valid, N={len(yv):,}) ---")
    print(f"Mean={np.mean(yv):.6f}  Std={np.std(yv):.6f}")
    print(f"Min={np.min(yv):.6f}  Max={np.max(yv):.6f}")
    print(f"Skew={float(pd_skew(yv)):.2f}  Kurt={float(pd_kurt(yv)):.2f}")
    print(f"Zero-cross={np.mean(yv > 0):.4f}")

    yiv = target_vals[target_valid_flags == 0]
    if len(yiv) > 0:
        print(f"\n--- Target stats (invalid/cross-session, N={len(yiv):,}) ---")
        print(f"Mean={np.mean(yiv):.6f}  Std={np.std(yiv):.6f}")


def pd_skew(x: np.ndarray) -> float:
    m2 = np.mean((x - x.mean()) ** 2)
    m3 = np.mean((x - x.mean()) ** 3)
    return m3 / (m2 ** 1.5) if m2 > 0 else 0.0


def pd_kurt(x: np.ndarray) -> float:
    m2 = np.mean((x - x.mean()) ** 2)
    m4 = np.mean((x - x.mean()) ** 4)
    return m4 / (m2 ** 2) - 3 if m2 > 0 else 0.0


if __name__ == "__main__":
    generate_windows()
