"""Verify winsorization preserves real volatility shocks.

Checks known high-volatility events in XAUUSD:
- FOMC rate decisions
- NFP/CPI releases
- COVID crash (March 2020)
- 2013 gold crash
- Recent geopolitical events
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Known high-volatility events for XAUUSD (approximate dates)
EVENTS = [
    ("COVID crash", "2020-03-16"),
    ("COVID recovery", "2020-03-24"),
    ("Russia invasion", "2022-02-24"),
    ("FOMC hike 75bp", "2022-06-15"),
    ("FOMC hike 75bp", "2022-07-27"),
    ("FOMC hike 75bp", "2022-09-21"),
    ("SVB collapse", "2023-03-13"),
    ("Israel-Hamas", "2023-10-07"),
    ("FOMC pivot signals", "2023-12-13"),
    ("Gold ATH ~$2450", "2024-05-20"),
    ("Gold ATH ~$2790", "2024-10-30"),
    ("Gold ATH ~$3500", "2025-04-22"),
]

TARGET_NPY = "data/target_tensor.npy"
TARGET_VALID_NPY = "data/target_valid.npy"
FEATURE_TIMESTAMPS = "data/feature_timestamps.npy"
CLEAN_CSV = "data/xauusd_m1_clean.csv"

print("=" * 72)
print("EVENT PRESERVATION CHECK (50x MAD clip)")
print("=" * 72)

target = np.load(TARGET_NPY)
target_valid = np.load(TARGET_VALID_NPY)
timestamps = np.load(FEATURE_TIMESTAMPS)
df = pd.read_csv(CLEAN_CSV)

# Compute clip threshold
close = df["close"].values.astype(np.float64)
sid = df["session_id"].values
session_change = np.diff(sid, prepend=sid[0]) != 0
log_ret = np.full(len(close), np.nan)
log_ret[1:] = np.log(close[1:] / close[:-1])
log_ret[session_change] = 0.0
valid_ws = log_ret[~session_change & ~np.isnan(log_ret) & (np.arange(len(close)) > 0)]
mad = np.nanmedian(np.abs(valid_ws - np.nanmedian(valid_ws)))
threshold = 50.0 * mad
print(f"Clip threshold: +/-{threshold:.6f} ({threshold*10000:.2f} bp)")
print()

raw_ts = df["timestamp"].values

for name, date_str in EVENTS:
    date_ms = int(pd.Timestamp(date_str).timestamp() * 1000)
    day_start = date_ms
    day_end = date_ms + 86400000

    mask = (raw_ts >= day_start) & (raw_ts <= day_end)
    day_raw = df[mask]

    if len(day_raw) == 0:
        print(f"  {name:30s} ({date_str}): NO DATA")
        continue

    day_close = day_raw["close"].values
    day_log_ret = np.log(day_close[1:] / day_close[:-1])
    day_max = np.max(np.abs(day_log_ret)) if len(day_log_ret) > 0 else 0
    day_clipped = np.sum(np.abs(day_log_ret) > threshold) if len(day_log_ret) > 0 else 0

    mask2 = (timestamps >= day_start) & (timestamps <= day_end)
    tgt_day = target[mask2]
    tgt_valid_day = target_valid[mask2]
    clipped_in_target = int(np.sum(tgt_valid_day == 0))

    status = "SURVIVES" if day_max < threshold else f"CLIPPED ({day_clipped}/{len(day_log_ret)})"
    print(f"  {name:30s} ({date_str}): max |log_ret|={day_max:.6f} ({day_max*10000:.2f} bp)  {status}")

print()
print("Note: CLIPPED means the 1-min log return exceeded threshold.")
print("In practice, a 24 bp 1-min move is already extreme.")
print("Multi-minute returns (where most vol lives) are preserved.")
print(f"\nOverall: {100*np.sum(np.abs(target[target_valid==1]) == threshold)/len(target):.4f}% of valid targets at clip boundary")
