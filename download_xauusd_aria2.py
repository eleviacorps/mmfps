"""Download all XAUUSD M1 BI5 files using aria2c for maximum parallelism."""
from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ARIA2C = Path("tools") / "aria2c.exe"
INSTRUMENT = "XAUUSD"
BI5_CACHE = Path("data") / "bi5_cache"
URLS_FILE = Path("data") / "bi5_urls.txt"

URL_TEMPLATE = "https://datafeed.dukascopy.com/datafeed/{instrument}/{year}/{month:02d}/{day:02d}/BID_candles_min_1.bi5"

# XAUUSD M1 data available since 2003-05-05
START_DATE = datetime(2003, 5, 5, tzinfo=timezone.utc)
TODAY = datetime.now(timezone.utc)
# Yesterday - today's data isn't available yet
LAST_DATE = TODAY - timedelta(days=1)


def generate_urls():
    """Generate aria2c input file with unique output names."""
    lines = []
    d = START_DATE
    count = 0
    while d <= LAST_DATE:
        url = URL_TEMPLATE.format(
            instrument=INSTRUMENT,
            year=d.year,
            month=d.month - 1,
            day=d.day,
        )
        fname = f"xauusd_{d.year}-{d.month:02d}-{d.day:02d}.bi5"
        lines.append(url)
        lines.append(f"  out={fname}")
        lines.append(f"  dir={BI5_CACHE.as_posix()}")
        d += timedelta(days=1)
        count += 1
    BI5_CACHE.mkdir(parents=True, exist_ok=True)
    URLS_FILE.parent.mkdir(parents=True, exist_ok=True)
    URLS_FILE.write_text("\n".join(lines))
    print(f"Generated {count} URLs -> {URLS_FILE}")
    print(f"First: {lines[0]}")
    print(f"Last:  {lines[-3]}")
    return count


def download():
    """Run aria2c with maximum parallelism."""
    cmd = [
        str(ARIA2C),
        "-i", str(URLS_FILE),
        "--max-concurrent-downloads=50",
        "--split=1",
        "--max-connection-per-server=4",
        "--connect-timeout=10",
        "--timeout=30",
        "--max-tries=3",
        "--retry-wait=1",
        "--auto-save-interval=10",
        "--console-log-level=notice",
        "--summary-interval=2",
        "--allow-overwrite=false",
        "--auto-file-renaming=false",
        "--continue=true",
    ]
    print(f"Running: {' '.join(cmd)}")
    print(f"Downloading to: {BI5_CACHE}")
    sys.stdout.flush()
    result = subprocess.run(cmd)
    if result.returncode == 0:
        print("\nAll downloads complete!")
    else:
        print(f"\naria2c exited with code {result.returncode} (some may have failed)")
    return result.returncode


def parse_all_bi5():
    """Parse all downloaded BI5 files into a single CSV."""
    import lzma
    import struct

    import numpy as np
    import pandas as pd

    bi5_files = sorted(BI5_CACHE.glob("xauusd_*.bi5"))
    print(f"Parsing {len(bi5_files)} BI5 files...")

    chunks = []
    dt = np.dtype([
        ("sec", ">i4"),
        ("open_raw", ">i4"),
        ("close_raw", ">i4"),
        ("low_raw", ">i4"),
        ("high_raw", ">i4"),
        ("volume", ">f4"),
    ])

    for fpath in bi5_files:
        compressed = fpath.read_bytes()
        if len(compressed) < 10:
            continue
        try:
            decompressed = lzma.decompress(compressed, format=lzma.FORMAT_ALONE)
        except lzma.LZMAError:
            continue

        n = len(decompressed) // 24
        if n == 0:
            continue

        # Parse date from filename: xauusd_2026-05-18.bi5
        parts = fpath.stem.split("_")
        date_str = parts[1]
        dt_obj = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        day_start_ms = int(dt_obj.timestamp() * 1000)

        data = np.frombuffer(decompressed[:n * 24], dtype=dt)
        timestamps = day_start_ms + data["sec"].astype(np.int64) * 1000
        arr = np.column_stack([
            timestamps,
            data["open_raw"] / 1000.0,
            data["high_raw"] / 1000.0,
            data["low_raw"] / 1000.0,
            data["close_raw"] / 1000.0,
            data["volume"],
        ])
        chunks.append(arr)

    if not chunks:
        print("No data parsed!")
        return None

    full = np.concatenate(chunks)
    full = full[np.argsort(full[:, 0])]

    df = pd.DataFrame(full, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = df["timestamp"].astype("int64")
    df = df.drop_duplicates(subset="timestamp").reset_index(drop=True)

    out_path = Path("data") / "xauusd_m1_raw.csv"
    df.to_parquet(str(out_path).replace(".csv", ".parquet"), index=False)
    df.to_csv(out_path, index=False)

    print(f"\nSaved {len(df):,} bars")
    print(f"Range: {pd.to_datetime(df['timestamp'].iloc[0], unit='ms')} "
          f"to {pd.to_datetime(df['timestamp'].iloc[-1], unit='ms')}")
    print(f"Prices: {df['low'].min():.3f} - {df['high'].max():.3f}")
    print(f"Volume: {df['volume'].min():.6f} - {df['volume'].max():.6f}")
    return df


def main():
    n = generate_urls()
    print(f"Prepared {n} URLs for aria2c")
    rc = download()
    # 0 = all OK, 3 = partial success (some 404s for weekends/holidays)
    if rc in (0, 3):
        parse_all_bi5()
    else:
        print("Downloads failed, skipping parse")


if __name__ == "__main__":
    main()
