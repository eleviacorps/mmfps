"""Test dukascopy-python for XAUUSD download."""
from datetime import datetime
import dukascopy_python

# Test a single day download
start = datetime(2026, 5, 18)
end = datetime(2026, 5, 19)
instrument = dukascopy_python.INSTRUMENT_FX_COMMODITIES_XAU_USD
interval = dukascopy_python.INTERVAL_MINUTE_1
offer_side = dukascopy_python.OFFER_SIDE_BID

print(f"Downloading XAUUSD from {start} to {end}...")
df = dukascopy_python.fetch(instrument, interval, offer_side, start, end)
print(f"Rows: {len(df)}")
print(df.head())
print(df.tail())
print(f"Columns: {list(df.columns)}")
print("Done")
