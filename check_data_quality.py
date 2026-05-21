"""Quick data quality check on downloaded XAUUSD data."""
import pandas as pd

df = pd.read_csv("data/xauusd_m1_raw.csv")
ts = pd.to_datetime(df["timestamp"], unit="ms")
diffs = ts.diff().dropna()

print(f"Bars: {len(df):,}")
print(f"Date range: {ts.iloc[0]} to {ts.iloc[-1]}")
print(f"Price range: low={df['low'].min():.3f} high={df['high'].max():.3f}")
print(f"Close: mean={df['close'].mean():.3f} std={df['close'].std():.3f}")
print(f"Volume: min={df['volume'].min():.6f} max={df['volume'].max():.6f}")
print(f"Zero volume: {(df['volume'] == 0).sum():,}")
print(f"NaN count: {df.isna().sum().sum()}")
print(f"Mean interval: {diffs.mean()}")
print(f"Max interval: {diffs.max()}")
print(f"Intervals > 2min: {(diffs > pd.Timedelta(minutes=2)).sum():,}")
print(f"Intervals > 1hr: {(diffs > pd.Timedelta(hours=1)).sum():,}")
