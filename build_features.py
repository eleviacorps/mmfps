"""SCRIPT 3: Build 32-64 stable features from clean XAUUSD M1 data.

Feature categories:
- Log returns (1, 5, 15, 60, 240 min)
- Rolling volatility (5, 15, 60, 240)
- RSI (14, 30, 60)
- MACD (12/26/9, 24/52/18)
- ATR (14, 30)
- Momentum (1, 5, 15, 60, 240)
- Price vs SMA ratio (20, 50, 200)
- Rolling skew/kurtosis (60, 240)
- Volume z-score (5, 15, 60)
- Autocorrelation of returns (lag 1, 5, 15)
- Session-aware features
- Target: price delta (close_{t+1} - close_t)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("data")
CLEAN_CSV = DATA_DIR / "xauusd_m1_clean.csv"
FEATURE_NPY = DATA_DIR / "feature_tensor.npy"
TARGET_NPY = DATA_DIR / "target_tensor.npy"
FEATURE_COLUMNS = DATA_DIR / "feature_columns.txt"

WINDOW = 256  # Rolling window size


def build_features(df: pd.DataFrame, close: np.ndarray, volume: np.ndarray,
                   high: np.ndarray, low: np.ndarray, session_boundary: np.ndarray) -> pd.DataFrame:
    n = len(df)
    feat = pd.DataFrame(index=df.index)

    # --- Log returns (always positive for positive prices) ---
    log_ret = np.full(n, np.nan)
    log_ret[1:] = np.log(close[1:] / close[:-1])
    log_ret[1:] = np.where(session_boundary[1:] == 1, 0, log_ret[1:])

    for lag, name in [(1, "1m"), (5, "5m"), (15, "15m"), (60, "1h"), (240, "4h")]:
        feat[f"log_ret_{name}"] = pd.Series(log_ret).rolling(lag).sum()

    # --- Rolling volatility (std of log returns) ---
    for window, name in [(5, "5"), (15, "15"), (60, "60"), (240, "240")]:
        feat[f"volatility_{name}"] = pd.Series(log_ret).rolling(window).std().fillna(0)

    # --- RSI ---
    def _rsi(series: np.ndarray, period: int) -> np.ndarray:
        delta = np.diff(series, prepend=series[0])
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gains).rolling(period).mean()
        avg_loss = pd.Series(losses).rolling(period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return (100 - 100 / (1 + rs)).fillna(50).values

    for period, name in [(14, "14"), (30, "30"), (60, "60")]:
        feat[f"rsi_{name}"] = _rsi(close, period)

    # --- MACD ---
    def _ema(data: np.ndarray, period: int) -> np.ndarray:
        alpha = 2 / (period + 1)
        out = np.full_like(data, np.nan)
        out[0] = data[0]
        for i in range(1, len(data)):
            out[i] = data[i] * alpha + out[i - 1] * (1 - alpha)
        return out

    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd = ema12 - ema26
    signal = _ema(macd, 9)
    feat["macd"] = macd
    feat["macd_signal"] = signal
    feat["macd_hist"] = macd - signal

    # Double MACD
    ema24 = _ema(close, 24)
    ema52 = _ema(close, 52)
    macd2 = ema24 - ema52
    signal2 = _ema(macd2, 18)
    feat["macd2"] = macd2
    feat["macd2_signal"] = signal2

    # --- ATR ---
    def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
        tr = np.maximum(high[1:] - low[1:],
                        np.maximum(np.abs(high[1:] - close[:-1]),
                                   np.abs(low[1:] - close[:-1])))
        tr = np.concatenate([[tr[0]], tr])
        return pd.Series(tr).rolling(period).mean().values

    for period, name in [(14, "14"), (30, "30")]:
        feat[f"atr_{name}"] = _atr(high, low, close, period)

    # --- Momentum ---
    for lag, name in [(1, "1"), (5, "5"), (15, "15"), (60, "60"), (240, "240")]:
        feat[f"momentum_{name}"] = np.where(
            session_boundary, 0, close - np.roll(close, lag)
        )

    # --- Price vs SMA ratio ---
    for period, name in [(20, "20"), (50, "50"), (200, "200")]:
        sma = pd.Series(close).rolling(period).mean()
        feat[f"price_sma_{name}"] = close / sma.values - 1

    # --- Rolling skew and kurtosis via running moments (O(n)) ---
    log_ret_s = pd.Series(log_ret)

    # Compute on every 10th bar for speed, then forward-fill
    for window, name in [(60, "60"), (240, "240")]:
        skew_arr = np.full(n, np.nan)
        kurt_arr = np.full(n, np.nan)
        sample_idx = np.arange(window - 1, n, 10)
        for idx in sample_idx:
            chunk = log_ret[idx - window + 1:idx + 1]
            valid = chunk[~np.isnan(chunk)]
            if len(valid) > 2:
                m2 = np.mean((valid - valid.mean()) ** 2)
                m3 = np.mean((valid - valid.mean()) ** 3)
                skew_arr[idx] = m3 / (m2 ** 1.5) if m2 > 0 else 0
            if len(valid) > 3:
                m2 = np.mean((valid - valid.mean()) ** 2)
                m4 = np.mean((valid - valid.mean()) ** 4)
                kurt_arr[idx] = m4 / (m2 ** 2) - 3 if m2 > 0 else 0
        feat[f"skew_{name}"] = pd.Series(skew_arr).ffill().fillna(0)
        feat[f"kurt_{name}"] = pd.Series(kurt_arr).ffill().fillna(0)

    # --- Volume z-score ---
    vol_s = pd.Series(volume)
    for window, name in [(5, "5"), (15, "15"), (60, "60")]:
        mean_v = vol_s.rolling(window).mean()
        std_v = vol_s.rolling(window).std().replace(0, np.nan)
        feat[f"volume_z_{name}"] = ((volume - mean_v) / std_v).fillna(0).values

    # --- Autocorrelation via fast FFT-based method on sampled data ---
    def fast_autocorr(arr: np.ndarray, lag: int, window: int) -> np.ndarray:
        out = np.full_like(arr, np.nan)
        step = max(1, window // 100)
        sample_idx = range(window, len(arr), step)
        for i in sample_idx:
            chunk = arr[i - window:i]
            valid = chunk[~np.isnan(chunk)]
            if len(valid) > lag + 1:
                center = valid - valid.mean()
                var = np.sum(center ** 2)
                if var > 0:
                    ac = np.sum(center[:len(center) - lag] * center[lag:]) / var
                    out[i] = ac
        return out

    for lag, w, name in [(1, 60, "1_60"), (5, 60, "5_60"), (1, 240, "1_240")]:
        feat[f"autocorr_{name}"] = pd.Series(fast_autocorr(log_ret, lag, w)).ffill().fillna(0)

    # --- Session features ---
    feat["session_boundary"] = session_boundary.astype(int)
    sb = session_boundary.astype(bool)
    cum_not_sb = (~sb).cumsum()
    cum_not_sb_s = pd.Series(cum_not_sb)
    reset_at = cum_not_sb_s.where(pd.Series(sb))
    feat["bars_since_session_start"] = (cum_not_sb_s - reset_at.ffill().fillna(0)).astype(np.int32)

    # --- Price level features ---
    feat["log_price"] = np.log(close)
    feat["high_low_ratio"] = np.log(high / low)
    feat["close_open_ratio"] = np.log(close / df["open"].values)

    return feat


def main():
    df = pd.read_csv(CLEAN_CSV)
    print(f"Loaded {len(df):,} clean bars")

    close = df["close"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    volume = df["volume"].values.astype(np.float64)
    session_boundary = df["session_boundary"].values.astype(bool)

    features = build_features(df, close, volume, high, low, session_boundary)

    # Target: forward price delta (close_{t+1} - close_t)
    target = np.full(len(df), np.nan)
    target[:-1] = close[1:] - close[:-1]

    # Remove rows with NaN features (first window rows excluded)
    valid = features.dropna().index
    feature_tensor = features.loc[valid].values.astype(np.float32)
    target_tensor = target[valid].astype(np.float32)
    timestamps = df["timestamp"].values[valid]

    print(f"\nFeature tensor: {feature_tensor.shape}")
    print(f"Target tensor: {target_tensor.shape}")
    print(f"Features: {features.shape[1]}")

    # Save session boundary for the valid rows (1 = boundary, 0 = continuation)
    session_boundary_valid = session_boundary[valid].astype(np.int8)
    np.save(DATA_DIR / "session_boundary.npy", session_boundary_valid)

    # Save
    np.save(FEATURE_NPY, feature_tensor)
    np.save(TARGET_NPY, target_tensor)
    np.save(DATA_DIR / "feature_timestamps.npy", timestamps)

    with open(FEATURE_COLUMNS, "w") as f:
        for col in features.columns:
            f.write(col + "\n")

    print(f"Saved: {FEATURE_NPY}")
    print(f"Saved: {TARGET_NPY}")
    print(f"Saved columns: {FEATURE_COLUMNS}")

    # Summary stats
    print(f"\n--- Feature Summary ---")
    print(f"Target mean: {target_tensor.mean():.6f}  std: {target_tensor.std():.6f}")
    print(f"Target min: {target_tensor.min():.6f}  max: {target_tensor.max():.6f}")
    print(f"Target zero-cross: {(target_tensor > 0).sum()} up / {(target_tensor <= 0).sum()} down")


if __name__ == "__main__":
    main()
