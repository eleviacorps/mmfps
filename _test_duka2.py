"""Test dukascopy-python for XAUUSD download."""
from datetime import datetime
import dukascopy_python

# Test a single day
start = datetime(2026, 5, 18)
end = datetime(2026, 5, 19)
interval = dukascopy_python.INTERVAL_MIN_1
offer_side = dukascopy_python.OFFER_SIDE_BID

print(f"Downloading XAUUSD from {start} to {end}...")
df = dukascopy_python.fetch("xauusd", interval, offer_side, start, end, debug=True)
print(f"Rows: {len(df)}")
print(df.head())
print(df.tail())
print(f"Columns: {list(df.columns)}")
print(f"Open range: {df['open'].min():.2f} - {df['open'].max():.2f}")
print("Done")
