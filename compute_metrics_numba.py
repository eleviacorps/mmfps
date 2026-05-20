"""Numba-optimized structural metrics — computes all 8 metrics for ALL samples.

Key fix from V2: DTW is now computed for EVERY sample, not just the first 10.
Uses Numba parallel loops for performance.

Stage 2 of the pipeline: takes raw generator output (paths, targets) and
adds structural quality scores.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
from numba import njit, prange
from tqdm import tqdm


# ── Numba kernels ──────────────────────────────────────────────────────────

@njit(fastmath=True, cache=True, parallel=True)
def _compute_all_metrics(paths: np.ndarray, target: np.ndarray) -> dict:
    """Compute all structural metrics for a chunk.

    Args:
        paths:  (B, K, T) float32 — generated returns
        target: (B, T)    float32 — real returns

    All metrics computed in parallel over samples.
    """
    B, K, T = paths.shape

    # Output arrays
    mse_dist   = np.zeros((B, K), dtype=np.float32)
    dir_sim    = np.zeros((B, K), dtype=np.float32)
    vol_sim    = np.zeros((B, K), dtype=np.float32)
    cumret_sim = np.zeros((B, K), dtype=np.float32)
    turning_sim   = np.zeros((B, K), dtype=np.float32)
    volprof_sim   = np.zeros((B, K), dtype=np.float32)
    trend_sim     = np.zeros((B, K), dtype=np.float32)
    rhythm_sim    = np.zeros((B, K), dtype=np.float32)
    dtw_sim       = np.zeros((B, K), dtype=np.float32)

    for b in prange(B):
        tgt = target[b]

        for k in range(K):
            p = paths[b, k]

            # MSE
            sq_diff = 0.0
            for t in range(T):
                sq_diff += (p[t] - tgt[t]) ** 2
            mse_dist[b, k] = sq_diff / T

            # Directional
            dir_sim[b, k] = 1.0 if np.sign(tgt[-1] - tgt[0]) == np.sign(p[-1] - p[0]) else 0.0

            # Volatility
            vol_real = 0.0
            for t in range(T - 1):
                vol_real += (tgt[t + 1] - tgt[t]) ** 2
            vol_real = np.sqrt(vol_real / max(T - 1, 1))

            vol_gen = 0.0
            for t in range(T - 1):
                vol_gen += (p[t + 1] - p[t]) ** 2
            vol_gen = np.sqrt(vol_gen / max(T - 1, 1))
            vol_sim[b, k] = -abs(vol_gen - vol_real)

            # Cumulative return
            cumret_sim[b, k] = -abs((p[-1] - p[0]) - (tgt[-1] - tgt[0]))

            # Turning points
            tp_real = _count_turning_points(tgt)
            tp_gen  = _count_turning_points(p)
            turning_sim[b, k] = 1.0 - min(abs(tp_gen - tp_real), 1.0)

            # Volatility profile
            vp_real = _vol_profile(tgt)
            vp_gen  = _vol_profile(p)
            diff_sum = 0.0
            for wi in range(4):
                diff_sum += abs(vp_gen[wi] - vp_real[wi])
            volprof_sim[b, k] = 1.0 - min(diff_sum / 4.0, 1.0)

            # Trend phases
            tr_real = _trend_phases(tgt)
            tr_gen  = _trend_phases(p)
            tr_dot = 0.0
            tr_sum = 0.0
            for ti in range(3):
                tr_dot += tr_gen[ti] * tr_real[ti]
                tr_sum += tr_real[ti]
            trend_sim[b, k] = tr_dot / max(tr_sum, 1e-6)

            # Rhythm (4-segment means)
            rh_real = _segment_means(tgt, 4)
            rh_gen  = _segment_means(p, 4)
            rh_diff = 0.0
            for ri in range(4):
                rh_diff += abs(rh_gen[ri] - rh_real[ri])
            rhythm_sim[b, k] = 1.0 - min(rh_diff / 4.0, 1.0)

            # DTW
            dtw_sim[b, k] = _fast_dtw(tgt, p, window=5)

    return {
        "mse_distance": mse_dist,
        "directional_similarity": dir_sim,
        "volatility_similarity": vol_sim,
        "cumulative_return_similarity": cumret_sim,
        "turning_point_similarity": turning_sim,
        "volatility_profile_similarity": volprof_sim,
        "trend_phase_similarity": trend_sim,
        "temporal_rhythm_similarity": rhythm_sim,
        "dtw_similarity": dtw_sim,
    }


@njit(fastmath=True, cache=True)
def _count_turning_points(s: np.ndarray) -> float:
    n = len(s)
    if n < 3:
        return 0.0
    vol = np.std(s)
    if vol < 1e-8:
        return 0.0
    threshold = vol * 0.5
    count = 0
    prev_sign = 0
    for i in range(1, n - 1):
        diff = s[i + 1] - s[i]
        sign = 1 if diff > threshold else (-1 if diff < -threshold else 0)
        if sign != 0 and sign != prev_sign and prev_sign != 0:
            count += 1
        prev_sign = sign
    return float(count) / max(n - 1, 1)


@njit(fastmath=True, cache=True)
def _vol_profile(s: np.ndarray) -> np.ndarray:
    out = np.zeros(4, dtype=np.float32)
    T = len(s)
    windows = np.array([2, 5, 10, 20], dtype=np.int32)
    for wi in range(4):
        w = windows[wi]
        if T < w:
            out[wi] = np.std(s)
        else:
            total = 0.0
            for i in range(T - w + 1):
                seg = s[i : i + w]
                total += np.std(seg)
            out[wi] = total / (T - w + 1)
    return out


@njit(fastmath=True, cache=True)
def _trend_phases(s: np.ndarray) -> np.ndarray:
    out = np.zeros(3, dtype=np.float32)
    n = len(s)
    if n < 2:
        out[2] = 1.0
        return out
    diffs = np.diff(s)
    vol = np.std(diffs)
    thresh = max(vol * 0.25, 1e-8)
    up = down = 0.0
    for d in diffs:
        if d > thresh:
            up += 1.0
        elif d < -thresh:
            down += 1.0
    out[0] = up / (n - 1)
    out[1] = down / (n - 1)
    out[2] = 1.0 - out[0] - out[1]
    return out


@njit(fastmath=True, cache=True)
def _segment_means(s: np.ndarray, n_segments: int) -> np.ndarray:
    T = len(s)
    seg_size = T // n_segments
    if seg_size < 1:
        return np.full(n_segments, np.mean(s), dtype=np.float32)
    out = np.zeros(n_segments, dtype=np.float32)
    for i in range(n_segments):
        start = i * seg_size
        end = (i + 1) * seg_size if i < n_segments - 1 else T
        total = 0.0
        for j in range(start, end):
            total += s[j]
        out[i] = total / (end - start)
    return out


@njit(fastmath=True, cache=True)
def _fast_dtw(s1: np.ndarray, s2: np.ndarray, window: int = 5) -> float:
    """Sakoe-Chiba bounded Dynamic Time Warping.

    Returns similarity score in (0, 1] — higher is better.
    """
    n, m = len(s1), len(s2)
    if n == 0 or m == 0:
        return 0.0

    bound = max(window, abs(n - m))
    D = np.full((n, m), 1e10, dtype=np.float32)
    D[0, 0] = abs(s1[0] - s2[0])

    for i in range(1, min(n, bound + 1)):
        D[i, 0] = D[i - 1, 0] + abs(s1[i] - s2[0])
    for j in range(1, min(m, bound + 1)):
        D[0, j] = D[0, j - 1] + abs(s1[0] - s2[j])

    for i in range(1, n):
        for j in range(1, m):
            if abs(i - j) <= bound:
                D[i, j] = abs(s1[i] - s2[j]) + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])

    return 1.0 / (1.0 + D[n - 1, m - 1] / max(n, m))


# ── Chunk processing ────────────────────────────────────────────────────────

def process_chunk(input_path: str, output_path: str) -> None:
    """Compute metrics for one chunk and save to disk."""
    d = np.load(input_path, mmap_mode="r")
    idx = d["idx"][:].astype(np.int32)
    target = d["target"][:].astype(np.float32)
    paths = d["paths"][:].astype(np.float32)
    behaviors = d["behaviors"][:].astype(np.float32)
    del d

    print(f"  Computing metrics ({idx.shape[0]} samples × {paths.shape[1]} paths)...")
    metrics = _compute_all_metrics(paths, target)

    # Composite score (same weights as V2)
    mse_dist = metrics["mse_distance"]
    mse_norm = (mse_dist - mse_dist.min(axis=-1, keepdims=True)) / (
        mse_dist.max(axis=-1, keepdims=True) - mse_dist.min(axis=-1, keepdims=True) + 1e-8
    )
    composite = (
        0.15 * (1 - mse_norm)
        + 0.15 * metrics["turning_point_similarity"]
        + 0.20 * metrics["volatility_profile_similarity"]
        + 0.15 * metrics["trend_phase_similarity"]
        + 0.15 * metrics["temporal_rhythm_similarity"]
        + 0.20 * metrics["dtw_similarity"]
    )
    best_by_composite = np.argmax(composite, axis=-1).astype(np.int32)
    path_ranks = np.argsort(mse_dist, axis=1).astype(np.int32)

    print(f"  Saving to {output_path}...")
    np.savez_compressed(
        output_path,
        idx=idx,
        target=target,
        paths=paths,
        behaviors=behaviors,
        mse_distance=mse_dist,
        directional_similarity=metrics["directional_similarity"],
        volatility_similarity=metrics["volatility_similarity"],
        cumulative_return_similarity=metrics["cumulative_return_similarity"],
        turning_point_similarity=metrics["turning_point_similarity"],
        volatility_profile_similarity=metrics["volatility_profile_similarity"],
        trend_phase_similarity=metrics["trend_phase_similarity"],
        temporal_rhythm_similarity=metrics["temporal_rhythm_similarity"],
        dtw_similarity=metrics["dtw_similarity"],
        path_rank=path_ranks,
        composite_score=composite,
        best_by_composite=best_by_composite,
    )
    print("  Done!")


def compute_all(input_dir: str, output_dir: str) -> None:
    """Process all NPZ chunks in input_dir."""
    os.makedirs(output_dir, exist_ok=True)

    chunks = sorted([f for f in os.listdir(input_dir) if f.endswith(".npz")])
    if not chunks:
        raise FileNotFoundError(f"No .npz files found in {input_dir}")

    print(f"Processing {len(chunks)} chunks from {input_dir} → {output_dir}")

    for i, chunk_name in enumerate(tqdm(chunks, desc="Metrics")):
        in_path = os.path.join(input_dir, chunk_name)
        out_path = os.path.join(output_dir, chunk_name)
        if os.path.exists(out_path):
            print(f"  Skip {chunk_name} (exists)")
            continue
        try:
            process_chunk(in_path, out_path)
        except Exception as e:
            print(f"  ERROR {chunk_name}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute structural metrics with Numba")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()
    compute_all(args.input_dir, args.output_dir)