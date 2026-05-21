"""Adaptive loss weight scheduler for curriculum learning progression.

Replaces hard stage cutoffs with smooth sigmoid-based annealing.
Tracks manifold quality to enable automatic stage readiness detection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class LossWeightSchedule:
    """Per-step loss weights based on training progress."""
    weight_trend: float = 0.0
    weight_turning: float = 0.0
    weight_diversity: float = 0.0
    weight_latent_sensitivity: float = 0.0
    weight_manifold_spread: float = 0.0


class AdaptiveLossWeightScheduler:
    """Smooth loss weight annealing with stage readiness monitoring.

    Replaces hard stage boundaries (A→B→C) with smooth sigmoid progression.
    Detects when current stage is saturated and enables next stage naturally.

    Usage:
        scheduler = AdaptiveLossWeightScheduler(
            warmup_steps=500,
            total_steps=5000,
        )
        
        for step in range(total_steps):
            weights = scheduler.get_weights(step)
            config.weight_trend = weights.weight_trend
            # ... train ...
            
            if step % 100 == 0:
                scheduler.record_manifold_quality(path_spread, behavior_distance)
                ready = scheduler.is_ready_for_next_stage()
    """

    def __init__(
        self,
        warmup_steps: int = 500,
        total_steps: int = 5000,
        enable_adaptive_stages: bool = True,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.enable_adaptive_stages = enable_adaptive_stages

        # Manifold quality tracking
        self.manifold_quality_history = []
        self.max_history = 100

        # Stage tracking (if using adaptive progression)
        self.current_stage = 0  # 0=pure_denoise, 1=structure, 2=diversity, 3=full
        self.stage_start_step = 0

    def _sigmoid(self, x: float, k: float = 5.0) -> float:
        """Smooth sigmoid: 0 at x=-1, 1 at x=+1."""
        return 1.0 / (1.0 + math.exp(-k * x))

    def get_weights(self, global_step: int) -> LossWeightSchedule:
        """Compute loss weights for current training step.

        Args:
            global_step: Current training step (0-indexed)

        Returns:
            LossWeightSchedule with per-component weights
        """
        if global_step < self.warmup_steps:
            # Stage 0: Pure denoising (reconstruction only)
            return LossWeightSchedule(
                weight_trend=0.0,
                weight_turning=0.0,
                weight_diversity=0.0,
                weight_latent_sensitivity=0.0,
                weight_manifold_spread=0.0,
            )

        # Progress through training: 0 (at warmup) → 1 (at total_steps)
        progress = (global_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, progress)

        # ─── Stage-based progression with smooth sigmoid ──────────────

        # Trend: activate at 20% progress
        trend_weight = 0.1 * self._sigmoid(10 * (progress - 0.2))

        # Turning: activate at 30% progress
        turning_weight = 0.05 * self._sigmoid(10 * (progress - 0.3))

        # Diversity: activate at 40% progress
        diversity_weight = 0.01 * self._sigmoid(8 * (progress - 0.4))

        # Latent sensitivity: activate at 50% progress
        latent_sens_weight = 0.05 * self._sigmoid(8 * (progress - 0.5))

        # Manifold spread: activate at 60% progress
        manifold_weight = 0.02 * self._sigmoid(8 * (progress - 0.6))

        return LossWeightSchedule(
            weight_trend=trend_weight,
            weight_turning=turning_weight,
            weight_diversity=diversity_weight,
            weight_latent_sensitivity=latent_sens_weight,
            weight_manifold_spread=manifold_weight,
        )

    def record_manifold_quality(
        self,
        path_spread: float,
        behavior_distance: float,
        loss_total: Optional[float] = None,
    ) -> None:
        """Record manifold quality metrics for readiness detection.

        Args:
            path_spread: Endpoint std dev across paths
            behavior_distance: Mean pairwise behavior distance
            loss_total: Total loss value (for stability check)
        """
        self.manifold_quality_history.append({
            "path_spread": path_spread,
            "behavior_distance": behavior_distance,
            "loss_total": loss_total or 0.0,
        })

        if len(self.manifold_quality_history) > self.max_history:
            self.manifold_quality_history.pop(0)

    def is_ready_for_next_stage(self) -> bool:
        """Detect if current stage is saturated (ready to progress).

        Returns:
            True if manifold quality is stable and ready to advance.
        """
        if not self.enable_adaptive_stages or len(self.manifold_quality_history) < 20:
            return False

        recent = self.manifold_quality_history[-20:]

        # Metrics should be stable (low variance)
        spreads = [m["path_spread"] for m in recent]
        spread_cv = (sum((s - sum(spreads) / len(spreads)) ** 2 for s in spreads) / len(spreads)) ** 0.5 / (sum(spreads) / len(spreads) + 1e-6)

        # Ready if: spread is stable AND above threshold
        spread_stable = spread_cv < 0.15
        spread_high = sum(spreads) / len(spreads) > 0.1

        losses = [m["loss_total"] for m in recent if m["loss_total"] > 0]
        if losses:
            loss_cv = (sum((l - sum(losses) / len(losses)) ** 2 for l in losses) / len(losses)) ** 0.5 / (sum(losses) / len(losses) + 1e-6)
            loss_stable = loss_cv < 0.2
        else:
            loss_stable = True

        return spread_stable and spread_high and loss_stable

    def get_stage_summary(self) -> dict:
        """Get summary of current stage health."""
        if not self.manifold_quality_history:
            return {"status": "no_data"}

        recent = self.manifold_quality_history[-50:]
        spreads = [m["path_spread"] for m in recent]
        distances = [m["behavior_distance"] for m in recent]

        return {
            "current_stage": self.current_stage,
            "samples_recorded": len(self.manifold_quality_history),
            "recent_spread_mean": sum(spreads) / len(spreads),
            "recent_spread_std": (sum((s - sum(spreads) / len(spreads)) ** 2 for s in spreads) / len(spreads)) ** 0.5,
            "recent_behavior_distance_mean": sum(distances) / len(distances),
            "ready_for_next_stage": self.is_ready_for_next_stage(),
        }
