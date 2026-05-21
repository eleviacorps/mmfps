"""Investigate BI5 format by comparing raw bytes with known-correct CSV."""
import struct
import lzma
import urllib.request

# Known correct bar from CSV: May 18, 2026 00:00 UTC
# timestamp=1779062400000, open=4536.825, high=4542.315, low=4532.765, close=4533.395

# Download BI5 for the same day from Dukascopy
url = "https://datafeed.dukascopy.com/datafeed/XAUUSD/2026/04/18/BID_candles_min_1.bi5"
print(f"Fetching: {url}")

try:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    resp = urllib.request.urlopen(req)
    compressed = resp.read()
    print(f"Downloaded {len(compressed)} bytes (compressed)")

    decompressed = lzma.decompress(compressed, format=lzma.FORMAT_ALONE)
    print(f"Decompressed {len(decompressed)} bytes")
    print(f"Number of bars: {len(decompressed) // 20}")
    print(f"Remainder bytes: {len(decompressed) % 20}")

    # Parse first 5 bars
    fmt = ">i I I I I"  # signed int for time?, unsigned for prices/volume
    for i in range(min(5, len(decompressed) // 20)):
        offset = i * 20
        data = decompressed[offset:offset + 20]
        vals = struct.unpack(fmt, data)
        print(f"\nBar {i}: {[hex(v) for v in vals]}")
        print(f"  Raw: {vals}")
        if vals[0] < 0:
            print(f"  Time is negative?")
        else:
            print(f"  Time raw: {vals[0]} -> {vals[0] * 1000 if vals[0] < 1e12 else vals[0]}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
