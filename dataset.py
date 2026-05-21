"""Dataset with temporal train/val/test split from pre-built features.

Loads flat feature_tensor.npy and creates multi-horizon windows:
  - short_seq:  (short_horizon, feature_dim)  — end of context window
  - mid_seq:    (mid_horizon, feature_dim)    — end of context window
  - long_seq:   (long_horizon, feature_dim)   — full context window
  - target:     (path_horizon,)               — future log returns

Splits are time-based (by timestamp, not random).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from config import BehaviorGenConfig


@dataclass
class Sample:
    short_seq: torch.Tensor
    mid_seq: torch.Tensor
    long_seq: torch.Tensor
    target: torch.Tensor
    sample_idx: int


class PathDataset(Dataset):
    """Memory-mapped dataset that extracts multi-horizon windows from pre-built features."""

    def __init__(
        self,
        config: BehaviorGenConfig | None = None,
        split: Literal["train", "val", "test"] = "train",
    ):
        self.cfg = config or BehaviorGenConfig()
        data_dir = Path(self.cfg.data_dir)

        self.features = np.load(str(data_dir / self.cfg.features_file), mmap_mode="r")
        self.targets = np.load(str(data_dir / self.cfg.targets_file), mmap_mode="r")
        self.target_valid = np.load(str(data_dir / self.cfg.target_valid_file), mmap_mode="r")
        self.session_ids = np.load(str(data_dir / self.cfg.session_id_file), mmap_mode="r")
        self.timestamps = np.load(str(data_dir / self.cfg.timestamps_file), mmap_mode="r")

        total_len = len(self.features)
        h_long = self.cfg.long_horizon
        h_path = self.cfg.path_horizon
        num_starts = total_len - h_long - h_path + 1

        if num_starts <= 0:
            raise ValueError(
                f"Data too short ({total_len} rows). Need at least "
                f"{h_long + h_path} for one sample."
            )

        self.start_indices = _compute_valid_starts(
            timestamps=self.timestamps,
            session_ids=self.session_ids,
            target_valid=self.target_valid,
            h_long=h_long,
            h_path=h_path,
            val_split_ts=self.cfg.val_split_ts,
            test_split_ts=self.cfg.test_split_ts,
            split=split,
        )
        self.max_samples = len(self.start_indices)
        self.split = split

    def __len__(self) -> int:
        return self.max_samples

    def __getitem__(self, idx: int) -> Sample:
        actual_idx = int(self.start_indices[idx])
        h_short = self.cfg.short_horizon
        h_mid = self.cfg.mid_horizon
        h_long = self.cfg.long_horizon
        h_path = self.cfg.path_horizon

        context_end = actual_idx + h_long
        future_end = context_end + h_path

        long_feats = self.features[actual_idx:context_end]

        short_start = context_end - h_short
        mid_start = context_end - h_mid

        short_feats = self.features[short_start:context_end]
        mid_feats = self.features[mid_start:context_end]

        # Target: cumulative path_horizon log returns
        target_returns = self.targets[context_end:future_end].copy()
        target_scaled = target_returns * self.cfg.target_scale

        return Sample(
            short_seq=torch.from_numpy(short_feats.astype(np.float32)),
            mid_seq=torch.from_numpy(mid_feats.astype(np.float32)),
            long_seq=torch.from_numpy(long_feats.astype(np.float32)),
            target=torch.from_numpy(target_scaled.astype(np.float32)),
            sample_idx=actual_idx,
        )


def collate_fn(samples: list[Sample]) -> Sample:
    return Sample(
        short_seq=torch.stack([s.short_seq for s in samples]),
        mid_seq=torch.stack([s.mid_seq for s in samples]),
        long_seq=torch.stack([s.long_seq for s in samples]),
        target=torch.stack([s.target for s in samples]),
        sample_idx=torch.tensor([s.sample_idx for s in samples]),
    )


def _compute_valid_starts(
    timestamps: np.ndarray,
    session_ids: np.ndarray,
    target_valid: np.ndarray,
    h_long: int,
    h_path: int,
    val_split_ts: int,
    test_split_ts: int,
    split: Literal["train", "val", "test"],
) -> np.ndarray:
    """Return start indices for session-pure, temporally split samples.

    A valid sample must keep the entire context and full future trajectory
    inside one market session, and every future target return must be marked
    valid. The split is assigned by the final timestamp of the future path.
    """
    total_len = len(timestamps)
    num_starts = total_len - h_long - h_path + 1
    starts = np.arange(num_starts, dtype=np.int64)
    context_end = starts + h_long
    future_end = context_end + h_path
    final_idx = future_end - 1

    val_cutoff = val_split_ts * 1000
    test_cutoff = test_split_ts * 1000
    final_ts = timestamps[final_idx]

    if split == "train":
        split_mask = final_ts < val_cutoff
    elif split == "val":
        split_mask = (final_ts >= val_cutoff) & (final_ts < test_cutoff)
    elif split == "test":
        split_mask = final_ts >= test_cutoff
    else:
        raise ValueError(f"Unknown split: {split!r}")

    same_session = session_ids[starts] == session_ids[final_idx]

    invalid = (target_valid == 0).astype(np.int32)
    invalid_cum = np.concatenate(([0], np.cumsum(invalid)))
    future_invalid_count = invalid_cum[future_end] - invalid_cum[context_end]
    future_targets_valid = future_invalid_count == 0

    valid = split_mask & same_session & future_targets_valid
    return starts[valid]


def build_splits(
    config: BehaviorGenConfig | None = None,
) -> tuple[PathDataset, PathDataset, PathDataset]:
    cfg = config or BehaviorGenConfig()

    train = PathDataset(cfg, split="train")

    target_chunks = []
    h_long = cfg.long_horizon
    h_path = cfg.path_horizon
    for chunk in np.array_split(train.start_indices, max(1, min(256, len(train.start_indices) // 50000 + 1))):
        if len(chunk) == 0:
            continue
        idx = chunk[:, None] + h_long + np.arange(h_path)[None, :]
        target_chunks.append(np.asarray(train.targets[idx], dtype=np.float32).reshape(-1))
    train_targets = np.concatenate(target_chunks) if target_chunks else np.array([], dtype=np.float32)
    train_std = float(np.std(train_targets))
    cfg.target_scale = float(1.0 / max(train_std, 1e-10))
    print(
        f"[build_splits] valid_train_windows={len(train):,}, "
        f"target_std={train_std:.8f}, target_scale={cfg.target_scale:.2f}"
    )

    val = PathDataset(cfg, split="val")
    test = PathDataset(cfg, split="test")
    return train, val, test
