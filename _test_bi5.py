import lzma, struct
import numpy as np
import requests

url = 'https://datafeed.dukascopy.com/datafeed/XAUUSD/2026/04/20/BID_candles_min_1.bi5'
resp = requests.get(url, timeout=30)
print(f'Status: {resp.status_code}, Size: {len(resp.content)} bytes')

if resp.status_code == 200:
    raw = lzma.decompress(resp.content)
    fmt = struct.Struct('>IiiiI')
    rec_bytes = fmt.size
    n = len(raw) // rec_bytes
    print(f'Records: {n}')
    
    arr = np.frombuffer(raw, dtype=np.dtype([
        ('time','>u4'),('open','>i4'),('high','>i4'),
        ('low','>i4'),('close','>i4'),('volume','>i4')
    ]), count=n)
    
    scale = 100000.0
    print('First 3:')
    for i in range(3):
        r = arr[i]
        print(f'  t={r["time"]} O={r["open"]/scale:.2f} H={r["high"]/scale:.2f} L={r["low"]/scale:.2f} C={r["close"]/scale:.2f} V={r["volume"]}')
    print('Last 3:')
    for i in range(n-3, n):
        r = arr[i]
        print(f'  t={r["time"]} O={r["open"]/scale:.2f} H={r["high"]/scale:.2f} L={r["low"]/scale:.2f} C={r["close"]/scale:.2f} V={r["volume"]}')

print('Done')
