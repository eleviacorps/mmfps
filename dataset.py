"""Dataset with temporal train/val/test split for MMFPS_GEN_V2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from .config import BehaviorGenConfig


@dataclass
class Sample:
    """Single training sample — returned by PathDataset."""
    short_seq: torch.Tensor    # (short_horizon, feature_dim)
    mid_seq: torch.Tensor      # (mid_horizon, feature_dim)
    long_seq: torch.Tensor     # (long_horizon, feature_dim)
    target: torch.Tensor      # (path_horizon,)   future returns
    last_price: float          # Last known price (for cumulative transform)
    sample_idx: int            # Index in original array


def _build_feature_table(prices: np.ndarray, feature_dim: int) -> np.ndarray:
    """Build (T, feature_dim) derived feature matrix from 1D price series.

    Features (8 channels):
        0: price_zscore     — z-score normalized price over full series
        1: raw_return       — (p[t] - p[t-1]) / p[t-1]
        2: log_return       — log(p[t] / p[t-1])
        3: rolling_vol_5    — rolling std of returns (window=5)
        4: rolling_vol_20   — rolling std of returns (window=20)
        5: momentum_delta   — (price / SMA_20) - 1
        6: ema_diff         — (price / EMA_10) - 1
        7: zscore_return    — z-score of return over 20-bar lookback
    """
    T = len(prices)
    eps = 1e-8
    feats = np.zeros((T, feature_dim), dtype=np.float32)

    ret = np.zeros(T, dtype=np.float32)
    ret[1:] = (prices[1:] - prices[:-1]) / (np.abs(prices[:-1]) + eps)

    log_ret = np.zeros(T, dtype=np.float32)
    safe_ratio = np.abs(prices[1:] / (np.abs(prices[:-1]) + eps))
    log_ret[1:] = np.log(np.maximum(safe_ratio, eps))

    vol_5 = np.zeros(T, dtype=np.float32)
    vol_20 = np.zeros(T, dtype=np.float32)
    for t in range(1, T):
        vol_5[t] = np.std(ret[max(0, t - 4):t + 1])
        vol_20[t] = np.std(ret[max(0, t - 19):t + 1])

    sma_20 = np.zeros(T, dtype=np.float32)
    cum = np.cumsum(prices)
    for t in range(T):
        if t >= 19:
            s = (cum[t] - (cum[t - 20] if t >= 20 else 0)) / 20.0
            sma_20[t] = s

    momentum = np.zeros(T, dtype=np.float32)
    mask_20 = sma_20 > eps
    momentum[mask_20] = (prices[mask_20] / sma_20[mask_20]) - 1.0

    ema_ratio = np.zeros(T, dtype=np.float32)
    alpha = 2.0 / 11.0
    ema = prices[0]
    for t in range(T):
        ema = alpha * prices[t] + (1.0 - alpha) * ema
        if ema > eps:
            ema_ratio[t] = (prices[t] / ema) - 1.0

    zscore = np.zeros(T, dtype=np.float32)
    for t in range(1, T):
        seg = ret[max(0, t - 20):t + 1]
        if len(seg) > 1:
            m, s = float(np.mean(seg)), float(np.std(seg))
            if s > eps:
                zscore[t] = (ret[t] - m) / s

    price_z = (prices - float(np.mean(prices))) / (float(np.std(prices)) + eps)

    feats[:, 0] = price_z
    feats[:, 1] = ret
    feats[:, 2] = log_ret
    feats[:, 3] = vol_5
    feats[:, 4] = vol_20
    feats[:, 5] = momentum
    feats[:, 6] = ema_ratio
    feats[:, 7] = zscore

    return feats


class PathDataset(Dataset):
    """Memory-mapped dataset that extracts multi-horizon windows.

    Context sequences are enriched with 8 derived features (returns,
    volatility, momentum, etc.) for richer GRU conditioning.
    Target remains single-channel returns space.
    """

    def __init__(
        self,
        data_path: str,
        config: BehaviorGenConfig | None = None,
        split: Literal["train", "val", "test"] = "train",
        price_column: int = 0,
    ):
        self.cfg = config or BehaviorGenConfig()
        self.split = split
        self.price_col = price_column

        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        self.data = np.load(str(data_path), mmap_mode="r")
        total_len = len(self.data)

        prices = self.data[:, price_column].astype(np.float32)
        self.prices = prices
        self.features = _build_feature_table(prices, self.cfg.feature_dim)

        h_long = self.cfg.long_horizon
        h_path = self.cfg.path_horizon
        max_available = total_len - h_long - h_path

        if max_available <= 0:
            raise ValueError(
                f"Data too short ({total_len} rows). Need at least "
                f"{h_long + h_path} for one sample."
            )

        # Temporal split — no shuffling, strictly chronological
        train_end = int(max_available * self.cfg.train_split)
        val_end = int(max_available * (self.cfg.train_split + self.cfg.val_split))

        if split == "train":
            self.start_idx, self.end_idx = 0, train_end
            self.max_samples = min(self.cfg.max_samples, train_end) if self.cfg.max_samples else train_end
        elif split == "val":
            self.start_idx, self.end_idx = train_end, val_end
            self.max_samples = val_end - train_end
        elif split == "test":
            self.start_idx, self.end_idx = val_end, max_available
            self.max_samples = max_available - val_end
        else:
            raise ValueError(f"Unknown split: {split!r}")

    def __len__(self) -> int:
        return self.max_samples

    def __getitem__(self, idx: int) -> Sample:
        actual_idx = self.start_idx + idx
        h_short = self.cfg.short_horizon
        h_mid = self.cfg.mid_horizon
        h_long = self.cfg.long_horizon
        h_path = self.cfg.path_horizon

        long_end = actual_idx + h_long
        path_end = long_end + h_path

        prices = self.prices

        long_prices = prices[actual_idx:long_end]
        last_price = float(long_prices[-1])

        short_end = long_end
        short_start = long_end - h_short
        mid_start = long_end - h_mid

        short_feats = self.features[short_start:short_end]
        mid_feats = self.features[mid_start:long_end]
        long_feats = self.features[actual_idx:long_end]

        target_prices = prices[long_end:path_end]

        returns = np.zeros(h_path, dtype=np.float32)
        lp = last_price
        for i in range(min(h_path, len(target_prices))):
            if abs(lp) > 1e-8:
                returns[i] = (target_prices[i] - lp) / (abs(lp) + 1e-8)
                lp = target_prices[i]

        return Sample(
            short_seq=torch.from_numpy(short_feats),
            mid_seq=torch.from_numpy(mid_feats),
            long_seq=torch.from_numpy(long_feats),
            target=torch.from_numpy(returns),
            last_price=float(target_prices[-1]) if len(target_prices) > 0 else last_price,
            sample_idx=actual_idx,
        )


def collate_fn(samples: list[Sample]) -> Sample:
    """Collate list of Sample objects into a batched Sample."""
    return Sample(
        short_seq=torch.stack([s.short_seq for s in samples]),
        mid_seq=torch.stack([s.mid_seq for s in samples]),
        long_seq=torch.stack([s.long_seq for s in samples]),
        target=torch.stack([s.target for s in samples]),
        last_price=torch.tensor([s.last_price for s in samples]),
        sample_idx=torch.tensor([s.sample_idx for s in samples]),
    )


def build_splits(
    data_path: str,
    config: BehaviorGenConfig | None = None,
) -> tuple[PathDataset, PathDataset, PathDataset]:
    """Build train/val/test datasets from the same data file.

    All splits share the same underlying memory-mapped array (copy-on-write
    is never triggered since we only read), so memory overhead is minimal.
    """
    cfg = config or BehaviorGenConfig()
    train = PathDataset(data_path, cfg, split="train")
    val = PathDataset(data_path, cfg, split="val")
    test = PathDataset(data_path, cfg, split="test")
    return train, val, test