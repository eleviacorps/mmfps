"""Lightweight metric tracking for diffusion training.

Phase 1: only track reconstruction loss and basic denoising diagnostics.
No pairwise cdist, no eigenvalue decompositions, no quantile on large tensors.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor


@dataclass
class MetricState:
    value: float = 0.0
    ema: float = 0.0


class MetricsTracker:
    """EMA-smoothed scalar metric tracking — no expensive tensor ops."""

    def __init__(self, ema_decay: float = 0.99):
        self.ema_decay = ema_decay
        self._metrics: dict[str, MetricState] = defaultdict(MetricState)
        self._step = 0

    def update(self, metrics: dict[str, float]) -> None:
        self._step += 1
        for key, val in metrics.items():
            state = self._metrics[key]
            state.value = val
            state.ema = state.ema * self.ema_decay + val * (1 - self.ema_decay)

    @staticmethod
    def compute_manifold_metrics(paths: Tensor, behaviors: Tensor) -> dict[str, float]:
        """Cheap diagnostics — endpoint std only, no pairwise."""
        endpoints = paths[:, :, -1]
        return {
            "manifold/path_spread": float(endpoints.std(dim=1).mean().item()),
            "manifold/endpoint_range": float(
                (endpoints.amax(dim=1) - endpoints.amin(dim=1)).mean().item()
            ),
        }

    @staticmethod
    def compute_structural_metrics(paths: Tensor, target: Tensor) -> dict[str, float]:
        """Cheap structural metrics — closest path distance + cone coverage."""
        B, K, T = paths.shape
        target_exp = target.unsqueeze(1) if target.ndim == 2 else target
        sq_diffs = ((paths - target_exp) ** 2).mean(dim=-1)
        min_dists = sq_diffs.min(dim=1).values
        real_final = target[:, -1]
        gen_final = paths[:, :, -1]
        in_cone = ((real_final >= gen_final.min(dim=1).values) &
                    (real_final <= gen_final.max(dim=1).values)).float()
        return {
            "structural/closest_distance": float(min_dists.mean().item()),
            "structural/mean_distance": float(sq_diffs.mean().item()),
            "structural/cone_coverage": float(in_cone.mean().item()),
        }

    def summary(self) -> dict[str, float]:
        return {k: v.ema for k, v in sorted(self._metrics.items())}

    def summary_str(self) -> str:
        parts = []
        for key in sorted(self._metrics.keys()):
            state = self._metrics[key]
            parts.append(f"{key.split('/')[-1]}={state.ema:.4f}")
        return " | ".join(parts)
