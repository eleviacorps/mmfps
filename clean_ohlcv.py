"""SCRIPT 2: Clean raw XAUUSD M1 OHLCV data.

Actions:
- Sort by timestamp, deduplicate
- Keep zero-volume bars (add liquidity markers instead of deleting)
- Add session_id, is_zero_volume, inactivity_duration
- Remove only truly invalid rows (negative prices, low>high)

Output: xauusd_m1_clean.csv + clean_report.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data")
RAW_CSV = DATA_DIR / "xauusd_m1_raw.csv"
CLEAN_CSV = DATA_DIR / "xauusd_m1_clean.csv"
CLEAN_REPORT = DATA_DIR / "clean_report.json"

SESSION_GAP_SECONDS = 300  # 5min gap = new trading session


def clean() -> pd.DataFrame:
    df = pd.read_csv(RAW_CSV)
    before = len(df)

    df = df.sort_values("timestamp").reset_index(drop=True)
    before_dedup = len(df)
    df = df.drop_duplicates(subset="timestamp")
    after_dedup = before_dedup - len(df)

    # Remove negative prices
    mask = (df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0)
    before_price = len(df)
    df = df[mask].reset_index(drop=True)
    removed_price = before_price - len(df)

    # Remove low > high violations
    before_lh = len(df)
    df = df[df["low"] <= df["high"]].reset_index(drop=True)
    removed_lh = before_lh - len(df)

    # --- Session detection ---
    ts = pd.to_datetime(df["timestamp"], unit="ms")
    gap_seconds = ts.diff().dt.total_seconds().fillna(0)
    session_start = gap_seconds > SESSION_GAP_SECONDS
    df["session_id"] = session_start.cumsum().astype(np.int32)

    # --- Liquidity markers (NOT removing zero-volume bars) ---
    df["is_zero_volume"] = (df["volume"] <= 0).astype(np.int8)

    # Inactivity duration (vectorized): seconds since last non-zero-volume bar
    ts_arr = df["timestamp"].values
    is_zv = df["is_zero_volume"].values.astype(bool)
    last_active = ts_arr.copy()
    for i in range(1, len(df)):
        if is_zv[i]:
            last_active[i] = last_active[i - 1]
        else:
            last_active[i] = ts_arr[i]
    inactivity = np.where(is_zv, (ts_arr - last_active) // 1000, 0)
    df["inactivity_duration"] = inactivity.astype(np.int64)

    # --- Report ---
    after = len(df)
    n_zero_vol = int(df["is_zero_volume"].sum())
    n_sessions = int(df["session_id"].nunique())
    gap_bounds = gap_seconds[session_start].describe()

    report = {
        "total_raw": before,
        "total_clean": after,
        "removed_duplicates": after_dedup,
        "removed_duplicate_pct": round(after_dedup / before * 100, 3),
        "removed_negative_price": removed_price,
        "removed_low_gt_high": removed_lh,
        "total_removed": before - after,
        "pct_retained": round(after / before * 100, 2),
        "zero_volume_bars": n_zero_vol,
        "zero_volume_pct": round(n_zero_vol / after * 100, 2),
        "sessions_detected": n_sessions,
        "gap_between_sessions_seconds": {
            "min": float(gap_bounds.get("min", 0)),
            "mean": float(gap_bounds.get("mean", 0)),
            "max": float(gap_bounds.get("max", 0)),
        },
    }

    print("=== CLEAN REPORT ===")
    print(f"Raw: {report['total_raw']:,}  Clean: {report['total_clean']:,}  "
          f"Removed: {report['total_removed']:,} ({100 - report['pct_retained']:.1f}%)")
    print(f"Sessions: {report['sessions_detected']:,}")
    print(f"Zero-volume bars: {report['zero_volume_bars']:,} ({report['zero_volume_pct']}%)")

    df.to_csv(CLEAN_CSV, index=False)
    print(f"Saved: {CLEAN_CSV}")

    with open(CLEAN_REPORT, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Report: {CLEAN_REPORT}")

    return df


if __name__ == "__main__":
    clean()
