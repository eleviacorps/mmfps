"""Fast XAUUSD M1 downloader using direct HTTP + BI5 parsing.

BI5 format (reverse-engineered from dukascopy-node v1.46.4):
- LZMA-alone compressed
- 24 bytes per record: >i I I I I f
  - int32: seconds within the day (UTC)
  - int32: open × 1000
  - int32: close × 1000
  - int32: low ÷ 1000
  - int32: high ÷ 1000
  - float32: volume (in millions)
"""
from __future__ import annotations

import lzma
import struct
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

INSTRUMENT = "XAUUSD"
DECIMAL_FACTOR = 1000
DATA_DIR = Path("data")
DOWNLOAD_DIR = DATA_DIR / "download"

URL_TEMPLATE = "https://datafeed.dukascopy.com/datafeed/{instrument}/{year}/{month:02d}/{day:02d}/BID_candles_min_1.bi5"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept-Encoding": "gzip, deflate",
}

CACHE_DIR = DOWNLOAD_DIR / ".bi5_cache"


def _date_range(start: datetime, end: datetime) -> Iterator[datetime]:
    """Iterate over all trading days between start and end (inclusive)."""
    d = start
    one_day = timedelta(days=1)
    while d <= end:
        yield d
        d += one_day


def download_day(date: datetime) -> np.ndarray | None:
    """Download and parse one day of XAUUSD M1 data. Returns array or None."""
    url = URL_TEMPLATE.format(
        instrument=INSTRUMENT,
        year=date.year,
        month=date.month - 1,  # Dukascopy uses 0-indexed months
        day=date.day,
    )

    cache_path = CACHE_DIR / f"{date.year}-{date.month:02d}-{date.day:02d}.npy"

    # Check cache first
    if cache_path.exists():
        return np.load(cache_path)

    try:
        req = Request(url, headers=HEADERS)
        resp = urlopen(req, timeout=30)
        compressed = resp.read()
    except Exception:
        return None  # No data for this day (weekend/holiday)

    if not compressed or len(compressed) < 10:
        return None

    try:
        decompressed = lzma.decompress(compressed, format=lzma.FORMAT_ALONE)
    except lzma.LZMAError:
        return None

    n_records = len(decompressed) // 24
    if n_records == 0:
        return None

    # Parse all records at once using numpy
    dt = np.dtype([
        ("sec", ">i4"),
        ("open_raw", ">i4"),
        ("close_raw", ">i4"),
        ("low_raw", ">i4"),
        ("high_raw", ">i4"),
        ("volume", ">f4"),
    ])
    data = np.frombuffer(decompressed[:n_records * 24], dtype=dt)

    # Compute timestamps: day_start_ms + sec * 1000
    day_start_ms = int(date.timestamp() * 1000)
    timestamps = day_start_ms + data["sec"].astype(np.int64) * 1000

    # Scale prices
    open_p = data["open_raw"] / DECIMAL_FACTOR
    close_p = data["close_raw"] / DECIMAL_FACTOR
    low_p = data["low_raw"] / DECIMAL_FACTOR
    high_p = data["high_raw"] / DECIMAL_FACTOR

    result = np.column_stack([
        timestamps,
        open_p,
        high_p,
        low_p,
        close_p,
        data["volume"],
    ])

    # Cache the result
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, result)

    return result


def download_range(start: datetime, end: datetime, max_workers: int = 10) -> list[np.ndarray]:
    """Download all days in [start, end] using parallel HTTP requests."""
    dates = list(_date_range(start, end))
    print(f"Downloading {len(dates)} days ({start.date()} to {end.date()}) "
          f"with {max_workers} workers...")

    results: list[np.ndarray] = []
    completed = 0
    errors = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(download_day, d): d for d in dates}
        for future in as_completed(futures):
            d = futures[future]
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
                    completed += 1
                else:
                    errors += 1
            except Exception:
                errors += 1

            if (completed + errors) % 100 == 0 or (completed + errors) == len(dates):
                elapsed = time.time() - t0
                print(f"  {completed + errors}/{len(dates)} days "
                      f"(ok={completed}, err={errors}, {elapsed:.0f}s)")

    print(f"Done: {completed} days with data, {errors} empty/error, "
          f"{time.time()-t0:.0f}s total")
    return results


def merge_and_save(results: list[np.ndarray], output_path: str | Path) -> pd.DataFrame:
    """Merge all daily arrays into a sorted, deduplicated CSV."""
    if not results:
        print("No data to merge!")
        return pd.DataFrame()

    full = np.concatenate(results)
    full = full[np.argsort(full[:, 0])]  # sort by timestamp

    df = pd.DataFrame(full, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = df["timestamp"].astype("int64")
    df = df.drop_duplicates(subset="timestamp").reset_index(drop=True)

    # Validate
    print(f"\nMerged: {len(df):,} bars")
    print(f"Date range: {pd.to_datetime(df['timestamp'].iloc[0], unit='ms')} "
          f"to {pd.to_datetime(df['timestamp'].iloc[-1], unit='ms')}")
    print(f"Price range: {df['low'].min():.3f} - {df['high'].max():.3f}")
    print(f"Volume range: {df['volume'].min():.6f} - {df['volume'].max():.6f}")

    df.to_parquet(str(output_path).replace(".csv", ".parquet"), index=False)
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    return df


def main():
    today = datetime.now(timezone.utc)

    # Download year by year from 2026 backwards
    years = list(range(today.year, 2002, -1))
    all_results = []
    for year in years:
        start = datetime(year, 1, 1, tzinfo=timezone.utc)
        end = datetime(year, 12, 31, tzinfo=timezone.utc)
        if year == today.year:
            end = today

        results = download_range(start, end, max_workers=10)
        all_results.extend(results)

        # Save incremental checkpoint after each year
        if results:
            checkpoint = DOWNLOAD_DIR / f"xauusd_m1_{year}_checkpoint.parquet"
            yr_df = merge_and_save(results, checkpoint)

    # Final merge
    if all_results:
        output_path = DATA_DIR / "xauusd_m1_raw.csv"
        merge_and_save(all_results, output_path)
    else:
        print("No data downloaded!")


if __name__ == "__main__":
    main()
