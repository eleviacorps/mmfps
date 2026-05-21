"""SCRIPT 2: Clean raw XAUUSD M1 OHLCV data.

Actions:
- Sort by timestamp
- Remove zero-volume bars (stale overnight quotes)
- Remove duplicate timestamps
- Remove invalid rows (negative prices, low>high)
- Save cleaned data + create session boundary markers for gaps

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


def clean() -> pd.DataFrame:
    df = pd.read_csv(RAW_CSV)

    before = len(df)

    # Sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Remove duplicate timestamps
    before_dedup = len(df)
    df = df.drop_duplicates(subset="timestamp")
    after_dedup = before_dedup - len(df)

    # Remove zero-volume bars
    before_vol = len(df)
    df = df[df["volume"] > 0].reset_index(drop=True)
    removed_vol = before_vol - len(df)

    # Remove negative prices
    mask = (df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0)
    before_price = len(df)
    df = df[mask].reset_index(drop=True)
    removed_price = before_price - len(df)

    # Remove low > high violations
    before_lh = len(df)
    df = df[df["low"] <= df["high"]].reset_index(drop=True)
    removed_lh = before_lh - len(df)

    after = len(df)

    # Mark session boundaries (gaps > 5min = trading session boundary)
    ts = pd.to_datetime(df["timestamp"], unit="ms")
    gap_seconds = ts.diff().dt.total_seconds()
    session_start = gap_seconds > 300  # 5min gap = new session
    df["session_boundary"] = session_start.astype(int)

    # Gap stats for report
    gaps_min = gap_seconds[session_start].describe()

    report = {
        "total_raw": before,
        "total_clean": after,
        "removed_duplicates": after_dedup,
        "removed_zero_volume": removed_vol,
        "removed_negative_price": removed_price,
        "removed_low_gt_high": removed_lh,
        "total_removed": before - after,
        "pct_retained": round(after / before * 100, 2),
        "sessions_detected": int(session_start.sum()),
        "gap_between_sessions_seconds": {
            "min": float(gaps_min.get("min", 0)),
            "mean": float(gaps_min.get("mean", 0)),
            "max": float(gaps_min.get("max", 0)),
        },
    }

    print("=== CLEAN REPORT ===")
    print(f"Raw: {report['total_raw']:,}  Clean: {report['total_clean']:,}  "
          f"Removed: {report['total_removed']:,} ({100 - report['pct_retained']:.1f}%)")
    print(f"Duplicates: {report['removed_duplicates']:,}")
    print(f"Zero volume: {report['removed_zero_volume']:,}")
    print(f"Negative price: {report['removed_negative_price']:,}")
    print(f"Low>High: {report['removed_low_gt_high']:,}")
    print(f"Trading sessions: {report['sessions_detected']:,}")

    df.to_csv(CLEAN_CSV, index=False)
    print(f"Saved: {CLEAN_CSV}")

    with open(CLEAN_REPORT, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Report: {CLEAN_REPORT}")

    return df


if __name__ == "__main__":
    clean()
