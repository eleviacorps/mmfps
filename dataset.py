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


class PathDataset(Dataset):
    """Memory-mapped dataset that extracts multi-horizon windows.

    The dataset operates on raw price data. It converts to returns space
    for the target, while context sequences stay as prices (expanded to
    feature_dim for model compatibility).
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

        self.last_price_col = h_long - 1  # index of last price in long sequence

    def __len__(self) -> int:
        return self.max_samples

    def __getitem__(self, idx: int) -> Sample:
        # Map to actual start of the window in the original array
        actual_idx = self.start_idx + idx
        h_short = self.cfg.short_horizon
        h_mid = self.cfg.mid_horizon
        h_long = self.cfg.long_horizon
        h_path = self.cfg.path_horizon

        price_col = self.price_col

        long_end = actual_idx + h_long
        path_end = long_end + h_path

        # Long sequence (480 bars up to prediction point)
        long_seq = self.data[actual_idx:long_end, price_col].astype(np.float32)

        # Last price before future — used to convert target back to price if needed
        last_price = float(long_seq[-1])

        # Short sequence: most recent h_short bars before prediction point
        short_start = long_end - h_short
        short_seq = self.data[short_start:long_end, price_col].astype(np.float32)

        # Mid sequence: h_mid bars ending at prediction point
        mid_start = long_end - h_mid
        mid_seq = self.data[mid_start:long_end, price_col].astype(np.float32)

        # Target: future prices (returns will be computed at loss time)
        target_prices = self.data[long_end:path_end, price_col].astype(np.float32)

        # Convert to returns: (P[t+1] - P[t]) / P[t]
        returns = np.zeros(h_path, dtype=np.float32)
        for i in range(h_path - 1):
            if last_price > 0 and abs(last_price) > 1e-8:
                returns[i] = (target_prices[i] - last_price) / (abs(last_price) + 1e-8)
                last_price = target_prices[i]
            else:
                returns[i] = 0.0
        # Last step
        if len(target_prices) > 1 and abs(target_prices[-2]) > 1e-8:
            returns[-1] = (target_prices[-1] - target_prices[-2]) / (abs(target_prices[-2]) + 1e-8)
        else:
            returns[-1] = 0.0

        # Expand context to (horizon, feature_dim) by broadcasting
        short_seq = short_seq[:, None]
        mid_seq = mid_seq[:, None]
        long_seq = long_seq[:, None]

        return Sample(
            short_seq=torch.from_numpy(short_seq),
            mid_seq=torch.from_numpy(mid_seq),
            long_seq=torch.from_numpy(long_seq),
            target=torch.from_numpy(returns),
            last_price=target_prices[-1] if len(target_prices) > 0 else last_price,
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