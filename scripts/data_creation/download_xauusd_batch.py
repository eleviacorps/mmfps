"""Batch XAUUSD M1 downloader using npx dukascopy-node.

Downloads year-by-year from recent to old. Starts working with
partial data as soon as first years complete.
"""
from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path("data")
DOWNLOAD_DIR = DATA_DIR / "download"
TODAY = datetime.now(timezone.utc)

# Download years from 2026 backwards to 2003
YEARS = list(range(2026, 2002, -1))


def download_year(year: int) -> Path | None:
    """Download one year of XAUUSD M1 data. Returns output path or None."""
    start = f"{year}-01-01"
    end = f"{year}-12-31"
    if year == 2026:
        end = TODAY.strftime("%Y-%m-%d")

    cmd = [
        "npx", "dukascopy-node",
        "-i", "xauusd",
        "-from", start,
        "-to", end,
        "-t", "m1",
        "-f", "csv",
        "-v",
        "-bs", "10",
        "-bp", "1500",
        "-dir", str(DOWNLOAD_DIR),
    ]

    print(f"\n{'='*60}")
    print(f"Downloading {year} ({start} to {end})...")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hours per year
        )
        print(result.stdout[-500:] if result.stdout else "")
        if result.returncode != 0:
            print(f"  WARNING: year {year} exited with code {result.returncode}")
            if result.stderr:
                print(f"  stderr: {result.stderr[-500:]}")
        else:
            print(f"  Year {year} complete!")
            return DOWNLOAD_DIR
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT: year {year} took too long")
    except Exception as e:
        print(f"  ERROR: year {year}: {e}")

    return None


def merge_all_years():
    """Concatenate all yearly CSVs into one sorted file."""
    import pandas as pd

    csv_files = sorted(DOWNLOAD_DIR.glob("xauusd-m1-bid-*.csv"))
    if not csv_files:
        print("No CSV files found to merge")
        return

    print(f"\nMerging {len(csv_files)} CSV files...")
    chunks = []
    for f in csv_files:
        print(f"  Reading {f.name}...")
        df = pd.read_csv(f)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        chunks.append(df)

    full = pd.concat(chunks, ignore_index=True)
    full = full.sort_values("timestamp").drop_duplicates(subset="timestamp")
    full = full.reset_index(drop=True)

    out_path = DATA_DIR / "xauusd_m1_raw.csv"
    full.to_csv(out_path, index=False)
    print(f"\nMerged: {len(full):,} bars")
    print(f"Date range: {full['timestamp'].min()} to {full['timestamp'].max()}")
    print(f"Saved: {out_path}")
    return full


def main():
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    successful = 0
    for year in YEARS:
        result = download_year(year)
        if result:
            successful += 1
        else:
            print(f"  Skipping year {year} after failure")

    print(f"\nDownloaded {successful}/{len(YEARS)} years")
    if successful > 0:
        merge_all_years()


if __name__ == "__main__":
    main()
