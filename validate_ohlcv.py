"""SCRIPT 1: Validate raw XAUUSD M1 OHLCV data.

Checks: missing timestamps, duplicates, NaNs, negative prices,
         intervals, outliers. Outputs validation_report.json."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data")
RAW_CSV = DATA_DIR / "xauusd_m1_raw.csv"
REPORT = DATA_DIR / "validation_report.json"


def validate() -> dict:
    df = pd.read_csv(RAW_CSV)
    ts = pd.to_datetime(df["timestamp"], unit="ms")
    df["ts"] = ts

    report: dict[str, object] = {}
    report["total_bars"] = len(df)
    report["date_range"] = {
        "start": str(ts.min()),
        "end": str(ts.max()),
    }

    # NaN check
    nan_count = df[["open", "high", "low", "close", "volume"]].isna().sum().to_dict()
    report["nan_count"] = nan_count
    report["total_nan"] = int(df.isna().sum().sum())

    # Negative prices
    neg_open = int((df["open"] <= 0).sum())
    neg_high = int((df["high"] <= 0).sum())
    neg_low = int((df["low"] <= 0).sum())
    neg_close = int((df["close"] <= 0).sum())
    report["negative_prices"] = {
        "open": neg_open,
        "high": neg_high,
        "low": neg_low,
        "close": neg_close,
    }

    # Duplicate timestamps
    dup_count = int(ts.duplicated().sum())
    report["duplicate_timestamps"] = dup_count

    # Monotonicity violations: low > high
    bad_low_high = int((df["low"] > df["high"]).sum())
    report["low_gt_high"] = bad_low_high

    # Interval analysis
    diffs = ts.sort_values().diff().dropna()
    intervals = diffs.dt.total_seconds()

    expected = 60  # M1 = 60 seconds
    gaps_2x = int((intervals > expected * 2).sum())
    gaps_1h = int((intervals > 3600).sum())
    gaps_1d = int((intervals > 86400).sum())
    max_gap = float(intervals.max())

    report["interval_stats"] = {
        "expected_seconds": expected,
        "mean_seconds": float(intervals.mean()),
        "median_seconds": float(intervals.median()),
        "std_seconds": float(intervals.std()),
        "max_gap_seconds": max_gap,
        "gaps_over_2min": gaps_2x,
        "gaps_over_1hr": gaps_1h,
        "gaps_over_1day": gaps_1d,
    }

    # Outlier detection on close
    close = df["close"].values
    q1, q3 = np.percentile(close, [25, 75])
    iqr = q3 - q1
    lower = q1 - 3 * iqr
    upper = q3 + 3 * iqr
    outliers = int(((close < lower) | (close > upper)).sum())

    report["outliers_iqr_3x"] = outliers
    report["close_stats"] = {
        "mean": float(close.mean()),
        "std": float(close.std()),
        "min": float(close.min()),
        "max": float(close.max()),
        "q1": float(q1),
        "q3": float(q3),
    }

    # Volume analysis
    vol = df["volume"].values
    zero_vol = int((vol == 0).sum())
    report["volume_zeros"] = zero_vol
    report["volume_pct_zero"] = round(zero_vol / len(df) * 100, 2)
    report["volume_stats"] = {
        "min": float(vol.min()),
        "max": float(vol.max()),
        "mean": float(vol.mean()),
        "median": float(np.median(vol)),
    }

    # Unique days with data
    unique_days = ts.dt.date.nunique()
    report["unique_trading_days"] = int(unique_days)

    print("=== VALIDATION REPORT ===")
    print(f"Total bars: {report['total_bars']:,}")
    print(f"Date range: {report['date_range']['start']} to {report['date_range']['end']}")
    print(f"NaN count: {report['total_nan']}")
    print(f"Duplicate timestamps: {report['duplicate_timestamps']}")
    print(f"Low > High violations: {report['low_gt_high']}")
    print(f"Max gap: {report['interval_stats']['max_gap_seconds']:.0f}s")
    print(f"Gaps over 1hr: {report['interval_stats']['gaps_over_1hr']:,}")
    print(f"Zero volume: {report['volume_zeros']:,} ({report['volume_pct_zero']}%)")
    print(f"Outliers (3x IQR): {report['outliers_iqr_3x']}")
    print(f"Unique trading days: {report['unique_trading_days']}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved: {REPORT}")

    return report


if __name__ == "__main__":
    validate()
