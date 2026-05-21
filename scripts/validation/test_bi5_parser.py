"""Test BI5 parser against known-good CSV data."""
from pathlib import Path

import pandas as pd

from scripts.data_creation.download_xauusd import download_day, merge_and_save
from datetime import datetime, timezone

# Test May 18, 2026 - we have verified CSV for this
test_date = datetime(2026, 5, 18, tzinfo=timezone.utc)
result = download_day(test_date)

if result is None:
    print("FAIL: No data for May 18, 2026")
else:
    print(f"Got {len(result)} bars")
    print(f"First bar: {result[0]}")

    # Read the known-good CSV
    csv_path = Path("data/download/xauusd-m1-bid-2026-05-18-2026-05-21.csv")
    csv_df = pd.read_csv(csv_path)
    csv_df["timestamp"] = pd.to_datetime(csv_df["timestamp"], unit="ms")
    may18 = csv_df[csv_df["timestamp"].dt.date == pd.Timestamp(test_date).date()]

    print(f"\nCSV has {len(may18)} bars for May 18")
    print(f"First CSV bar: {may18.iloc[0].values}")

    # Compare first bar
    our_first = result[0]
    csv_first = may18.iloc[0]
    ts_match = abs(our_first[0] - csv_first["timestamp"]) < 1000
    print(f"\nTimestamp match: {ts_match} (ours={our_first[0]}, csv={csv_first['timestamp']})")
    print(f"Price match: open={our_first[1]:.3f} vs {csv_first['open']:.3f}")

    # Parse and save full result for inspection
    full = merge_and_save([result], Path("data/test_bi5_output.csv"))
    print("\nFull comparison:")
    print(full.head())
