"""Diagnose target distribution: price deltas vs log returns, session effects."""
import numpy as np
import pandas as pd

df = pd.read_csv("data/xauusd_m1_clean.csv")
close = df["close"].values.astype(np.float64)
sb = df["session_boundary"].values.astype(bool)

# Price deltas (current targets)
price_delta = np.full(len(close), np.nan)
price_delta[:-1] = close[1:] - close[:-1]

# Log returns
log_ret = np.full(len(close), np.nan)
log_ret[:-1] = np.log(close[1:] / close[:-1])

# Session-aware: targets that cross session boundaries
cross_session = sb[1:]  # True where target spans a session boundary
within_session = ~cross_session

print("=== TARGET DISTRIBUTION DIAGNOSIS ===\n")

def describe(arr, label):
    valid = arr[~np.isnan(arr)]
    print(f"{label}:")
    print(f"  N={len(valid):,}  mean={np.mean(valid):.6f}  std={np.std(valid):.6f}")
    print(f"  min={np.min(valid):.6f}  max={np.max(valid):.6f}")
    print(f"  skew={pd.Series(valid).skew():.2f}  kurt={pd.Series(valid).kurtosis():.2f}")
    p99 = np.percentile(np.abs(valid), 99)
    p999 = np.percentile(np.abs(valid), 99.9)
    print(f"  99th pctile |val|: {p99:.6f}  99.9th: {p999:.6f}")
    print(f"  Zero-cross: {np.mean(valid > 0):.4f}")
    return valid

print("--- PRICE DELTAS ---")
pd_all = describe(price_delta, "All")
print()

print("--- PRICE DELTAS: Within-session ---")
pd_ws = price_delta[:-1][within_session]
pd_ws = pd_ws[~np.isnan(pd_ws)]
describe(pd_ws, "Within-session")
print()

print("--- PRICE DELTAS: Cross-session ---")
pd_cs = price_delta[:-1][cross_session]
pd_cs = pd_cs[~np.isnan(pd_cs)]
describe(pd_cs, "Cross-session")
print()

print("--- LOG RETURNS ---")
lr_all = describe(log_ret, "All")
print()

print("--- LOG RETURNS: Within-session ---")
lr_ws = log_ret[:-1][within_session]
lr_ws = lr_ws[~np.isnan(lr_ws)]
describe(lr_ws, "Within-session")
print()

print("--- LOG RETURNS: Cross-session ---")
lr_cs = log_ret[:-1][cross_session]
lr_cs = lr_cs[~np.isnan(lr_cs)]
describe(lr_cs, "Cross-session")
print()

# Winsorized log returns (within-session, clip at 10 sigma)
sigma = np.std(lr_ws)
lr_wins = np.clip(lr_ws, -10*sigma, 10*sigma)
print("--- LOG RETURNS: Within-session, winsorized 10σ ---")
describe(lr_wins, "Winsorized")
print(f"  Clipped: {np.sum(np.abs(lr_ws) > 10*sigma)} / {len(lr_ws)} ({(np.sum(np.abs(lr_ws) > 10*sigma) / len(lr_ws) * 100):.4f}%)")
