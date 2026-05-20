"""Real-time training metric tracking with exponential moving averages.

Tracks: losses, manifold metrics, structural similarity, diversity stats,
and distribution matching. All metrics are EMA-smoothed for stability.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor


@dataclass
class MetricState:
    """Holds latest value and EMA for one metric."""
    value: float = 0.0
    ema: float = 0.0


class MetricsTracker:
    """Collects and smooths training metrics across steps.

    Usage:
        tracker = MetricsTracker(ema_decay=0.99)
        tracker.update({
            "loss/total": 0.5,
            "loss/reconstruction": 0.4,
            "manifold/path_spread": 0.12,
        })
        print(tracker.summary())
    """

    def __init__(self, ema_decay: float = 0.99):
        self.ema_decay = ema_decay
        self._metrics: dict[str, MetricState] = defaultdict(MetricState)
        self._step = 0

    def update(self, metrics: dict[str, float]) -> None:
        self._step += 1
        for key, val in metrics.items():
            state = self._metrics[key]
            state.value = val
            state.ema = (
                state.ema * self.ema_decay + val * (1 - self.ema_decay)
            )

    @staticmethod
    def compute_manifold_metrics(
        paths: Tensor,
        behaviors: Tensor,
    ) -> dict[str, float]:
        """Compute manifold health metrics from generated paths.

        Args:
            paths:      (B, K, T) generated returns
            behaviors:  (B, K, D) per-path behavior embeddings

        Returns:
            dict with keys like "manifold/path_spread", "manifold/unique_ratio", etc.
        """
        B, K, T = paths.shape
        metrics = {}

        # Path spread: std of endpoint values across paths, averaged over batch
        endpoints = paths[:, :, -1]                       # (B, K)
        metrics["manifold/path_spread"] = float(endpoints.std(dim=1).mean().item())
        metrics["manifold/endpoint_range"] = float(
            (endpoints.amax(dim=1) - endpoints.amin(dim=1)).mean().item()
        )

        # Unique path ratio: how many paths are "unique" (pairwise distance > threshold)
        paths_2d = paths.reshape(B * K, T)
        dists = torch.cdist(paths_2d, paths_2d, p=2)
        mask = ~torch.eye(B * K, device=paths.device).bool()
        # Unique ≈ average distance large enough
        metrics["manifold/pairwise_distance"] = float(dists[mask].mean().item())
        metrics["manifold/pairwise_distance_std"] = float(dists[mask].std().item())

        # Future variance: how much total variance exists in generated futures
        metrics["manifold/future_variance"] = float(paths.var(dim=1).mean().item())

        # Behavior diversity
        _, _, D = behaviors.shape
        beh_flat = behaviors.reshape(-1, D)
        beh_dists = torch.cdist(beh_flat, beh_flat, p=2)
        beh_mask = ~torch.eye(B * K, device=paths.device).bool()
        metrics["manifold/behavior_diversity"] = float(beh_dists[beh_mask].mean().item())

        # Collapse check: if all paths have same sign on final step → collapsing
        final_signs = torch.sign(paths[:, :, -1])
        same_sign_pct = (final_signs.abs().sum(dim=1) / K).mean()
        metrics["manifold/sign_entropy"] = float(same_sign_pct.item())

        return metrics

    @staticmethod
    def compute_structural_metrics(
        paths: Tensor,
        target: Tensor,
    ) -> dict[str, float]:
        """Compute structural alignment metrics against real target.

        Args:
            paths:  (B, K, T) generated returns
            target: (B, T) real returns

        Returns:
            dict like "structural/coverage_rate", "structural/closest_dist", etc.
        """
        B, K, T = paths.shape
        metrics = {}

        if target.ndim == 2:
            target_exp = target.unsqueeze(1)                 # (B, 1, T)
        else:
            target_exp = target

        # Closest path distance (MSE space)
        sq_diffs = ((paths - target_exp) ** 2).mean(dim=-1)   # (B, K)
        min_dists = sq_diffs.min(dim=1).values                 # (B,)
        metrics["structural/closest_distance"] = float(min_dists.mean().item())
        metrics["structural/mean_distance"] = float(sq_diffs.mean().item())

        # Coverage: does real final price fall within generated range?
        real_final = target[:, -1]
        gen_final = paths[:, :, -1]
        gen_min = gen_final.min(dim=1).values
        gen_max = gen_final.max(dim=1).values
        in_cone = ((real_final >= gen_min) & (real_final <= gen_max)).float()
        metrics["structural/cone_coverage"] = float(in_cone.mean().item())

        # Directional accuracy of closest path
        real_dir = torch.sign(target[:, -1] - target[:, 0])
        gen_dir = torch.sign(paths[:, :, -1] - paths[:, :, 0])
        # Mean direction match across paths
        dir_match = (gen_dir == real_dir.unsqueeze(1)).float().mean()
        metrics["structural/direction_match"] = float(dir_match.item())

        # Top-K coverage (adaptive to actual K)
        K = sq_diffs.shape[1]
        for k in [1, 5, 10]:
            k_actual = min(k, K)
            best_k_idx = sq_diffs.topk(k_actual, dim=1, largest=False).indices
            best_k_dists = sq_diffs.gather(1, best_k_idx)
            metrics[f"structural/top_{k_actual}_dist"] = float(best_k_dists.mean().item())

        return metrics

    def summary(self) -> dict[str, float]:
        """Return all tracked metrics with current EMA values."""
        return {k: v.ema for k, v in sorted(self._metrics.items())}

    def summary_str(self) -> str:
        """One-line summary for tqdm / logging."""
        parts = []
        for key in sorted(self._metrics.keys()):
            state = self._metrics[key]
            parts.append(f"{key.split('/')[-1]}={state.ema:.4f}")
        return " | ".join(parts)