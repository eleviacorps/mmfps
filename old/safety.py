"""Global safety utilities for MMFPS_GEN_V2 training.

Provides centralized NaN/Inf detection, tensor sanitization, safe operations,
and emergency guards to prevent training crashes from numerical instability.

Design philosophy:
  - No training-critical path should operate without guards
  - Safety failures should degrade gracefully, not crash
  - Emergency skip logic prevents optimizer step on exploding gradients
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


# ── Core sanitization ────────────────────────────────────────────────────────

def sanitize(
    x: Tensor,
    fill_value: float = 0.0,
    clamp_range: tuple[float, float] | None = None,
    replace_nan: bool = True,
    replace_inf: bool = True,
) -> Tensor:
    """Replace NaN and Inf values in tensor with fill_value."""
    if replace_nan:
        x = torch.where(torch.isnan(x), torch.full_like(x, fill_value), x)
    if replace_inf:
        x = torch.where(torch.isinf(x), torch.full_like(x, fill_value), x)
    if clamp_range is not None:
        x = torch.clamp(x, *clamp_range)
    return x


def safe_divide(
    numerator: Tensor,
    denominator: Tensor,
    fallback: float = 0.0,
    epsilon: float = 1e-8,
) -> Tensor:
    """Safe division with fallback for zero denominator."""
    denom_safe = torch.where(
        denominator.abs() < epsilon,
        torch.full_like(denominator, epsilon),
        denominator,
    )
    return torch.where(
        torch.isfinite(numerator / denom_safe),
        numerator / denom_safe,
        torch.full_like(numerator, fallback),
    )


def safe_log(x: Tensor, floor: float = 1e-6) -> Tensor:
    """Log with NaN/Inf guard."""
    x_safe = x.clamp(min=floor)
    return torch.where(
        torch.isfinite(torch.log(x_safe)),
        torch.log(x_safe),
        torch.zeros_like(x),
    )


# ── Reduction safety ─────────────────────────────────────────────────────────

def safe_mean(x: Tensor, fallback: float = 0.0) -> float:
    """Mean with NaN/Inf guard."""
    val = x.mean()
    return float(val) if torch.isfinite(val) else fallback


def safe_std(x: Tensor, fallback: float = 0.0) -> float:
    """Std with NaN/Inf guard."""
    val = x.std()
    return float(val) if torch.isfinite(val) else fallback


# ── Tensor statistics ─────────────────────────────────────────────────────────

def tensor_stats(
    x: Tensor,
    prefix: str = "",
) -> dict[str, float]:
    """Compute comprehensive statistics for a tensor."""
    return {
        f"{prefix}mean": float(x.mean()),
        f"{prefix}std": float(x.std()),
        f"{prefix}min": float(x.min()),
        f"{prefix}max": float(x.max()),
        f"{prefix}max_abs": float(x.abs().max()),
        f"{prefix}nan_count": float(torch.isnan(x).sum()),
        f"{prefix}inf_count": float(torch.isinf(x).sum()),
        f"{prefix}is_finite": float(torch.all(torch.isfinite(x)).item()),
    }


def is_stable(x: Tensor) -> bool:
    """Check if tensor is stable (no NaN/Inf)."""
    return bool(torch.all(torch.isfinite(x)).item())


# ── Safe covariance ──────────────────────────────────────────────────────────

def safe_covariance(x: Tensor, eps: float = 1e-4) -> Tensor:
    """Compute covariance matrix with regularization.

    Uses eigenvalue floor to prevent singular matrices from producing NaN.
    """
    x_centered = x - x.mean(dim=0, keepdim=True)
    cov = x_centered.T @ x_centered / max(x.shape[0] - 1, 1)
    # Regularize to prevent singular logdet
    cov = cov + eps * torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
    return cov


def safe_logdet(x: Tensor, fallback: float = 0.0) -> float:
    """Safe log-determinant of a covariance matrix."""
    try:
        val = torch.logdet(x)
        return float(val) if torch.isfinite(val) else fallback
    except RuntimeError:
        eigvals = torch.linalg.eigvalsh(x)
        eigvals = eigvals.clamp(min=1e-6)
        return float(torch.log(eigvals).sum())


# ── Gradient safety ────────────────────────────────────────────────────────────

def clip_gradients(
    model: torch.nn.Module,
    max_norm: float = 1.0,
    eps: float = 1e-6,
) -> dict[str, float]:
    """Clip gradients by global norm, return clip stats."""
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    return {
        "grad_norm": float(total_norm) if not math.isinf(total_norm) and not math.isnan(total_norm) else 0.0,
        "grad_clipped": 1.0 if (not math.isinf(total_norm) and not math.isnan(total_norm) and total_norm > max_norm) else 0.0,
        "grad_nan": 0.0,
    }


def compute_gradient_health(
    model: torch.nn.Module,
) -> dict[str, float]:
    """Comprehensive gradient health analysis without modifying gradients.

    Returns multiple diagnostic metrics for gradient quality assessment.
    """
    metrics = {}
    param_groups = {}
    layer_norms = []
    activation_scales = []

    for name, param in model.named_parameters():
        if not param.requires_grad or param.grad is None:
            continue

        grad = param.grad
        has_nan = bool(torch.any(torch.isnan(grad)).item())
        has_inf = bool(torch.any(torch.isinf(grad)).item())

        # Layer-wise norm
        layer_norm = float(grad.norm().item())
        layer_norms.append(layer_norm)

        # Extract layer type from name
        layer_type = name.split('.')[0] if '.' in name else name
        if layer_type not in param_groups:
            param_groups[layer_type] = {'norm': 0.0, 'count': 0, 'nan': False, 'inf': False}

        param_groups[layer_type]['norm'] += layer_norm ** 2
        param_groups[layer_type]['count'] += 1
        param_groups[layer_type]['nan'] = param_groups[layer_type]['nan'] or has_nan
        param_groups[layer_type]['inf'] = param_groups[layer_type]['inf'] or has_inf

        # Activation scale (for ReLU/SiLU outputs)
        if param.data.dim() > 0:
            activation_scales.append(param.data.abs().mean().item())

    # Global statistics
    total_norm = math.sqrt(sum(ln ** 2 for ln in layer_norms)) if layer_norms else 0.0
    metrics["grad_total_norm"] = float(total_norm)
    metrics["grad_mean_norm"] = float(sum(layer_norms) / len(layer_norms)) if layer_norms else 0.0
    metrics["grad_max_norm"] = float(max(layer_norms)) if layer_norms else 0.0
    metrics["grad_min_norm"] = float(min(layer_norms)) if layer_norms else 0.0

    # Ratio of max to min (condition number indicator)
    if metrics["grad_min_norm"] > 1e-8:
        metrics["grad_condition_number"] = metrics["grad_max_norm"] / (metrics["grad_min_norm"] + 1e-8)
    else:
        metrics["grad_condition_number"] = float('inf')

    # Per-layer health
    for layer_type, group in sorted(param_groups.items()):
        group_norm = math.sqrt(group['norm'])
        metrics[f"grad_layer/{layer_type}/norm"] = float(group_norm)
        metrics[f"grad_layer/{layer_type}/has_nan"] = float(group['nan'])
        metrics[f"grad_layer/{layer_type}/has_inf"] = float(group['inf'])

    # Activation statistics
    if activation_scales:
        metrics["activation_mean_scale"] = float(sum(activation_scales) / len(activation_scales))
        metrics["activation_max_scale"] = float(max(activation_scales))
        metrics["activation_min_scale"] = float(min(activation_scales))

    return metrics


def check_gradients(
    model: torch.nn.Module,
) -> tuple[bool, dict[str, float]]:
    """Check if model gradients are stable. Returns (is_ok, stats)."""
    has_nan = False
    has_inf = False
    total_norm = 0.0
    param_count = 0

    for p in model.parameters():
        if not p.requires_grad:
            continue
        grad = p.grad
        if grad is None:
            continue
        has_nan = has_nan or bool(torch.any(torch.isnan(grad)))
        has_inf = has_inf or bool(torch.any(torch.isinf(grad)))
        total_norm += float(grad.norm().square())
        param_count += 1

    total_norm = total_norm ** 0.5
    is_ok = not has_nan and not has_inf and not math.isinf(total_norm) and not math.isnan(total_norm)

    return is_ok, {
        "grad_has_nan": float(has_nan),
        "grad_has_inf": float(has_inf),
        "grad_total_norm": float(total_norm) if not math.isinf(total_norm) and not math.isnan(total_norm) else 0.0,
        "grad_param_count": param_count,
    }


# ── Training safety ───────────────────────────────────────────────────────────

class TrainingSafetyMonitor:
    """Per-step monitoring for training stability.

    Tracks NaN events, skipped batches, and gradient explosions to
    provide visibility into training health.
    """

    def __init__(self, ema_decay: float = 0.95):
        self.nan_count = 0
        self.inf_count = 0
        self.skipped_batches = 0
        self.total_batches = 0
        self.ema_decay = ema_decay
        self.ema_grad_norm = 0.0  # Will learn from first valid step
        self.ema_loss = 0.0

    def record_batch(
        self,
        loss: Tensor | float,
        grad_norm: float,
        nan: bool = False,
        inf: bool = False,
    ) -> None:
        """Record a batch result."""
        self.total_batches += 1
        if nan:
            self.nan_count += 1
        if inf:
            self.inf_count += 1

        loss_val = float(loss) if isinstance(loss, Tensor) else loss
        self.ema_loss = self.ema_loss * self.ema_decay + loss_val * (1 - self.ema_decay)
        self.ema_grad_norm = self.ema_grad_norm * self.ema_decay + grad_norm * (1 - self.ema_decay)

    def record_skip(self) -> None:
        """Record a skipped optimizer step."""
        self.skipped_batches += 1

    def should_skip_step(self, grad_norm: float, threshold: float = 100.0) -> bool:
        """Determine if current gradient is too explosive to apply."""
        if torch.isnan(torch.tensor(grad_norm)) or torch.isinf(torch.tensor(grad_norm)):
            return True
        # Skip if gradient is 10x above EMA norm (explosion detection)
        if self.ema_grad_norm > 0 and grad_norm > threshold * max(self.ema_grad_norm, 1e-6):
            return True
        return False

    def summary(self) -> dict[str, float]:
        """Get stability summary."""
        return {
            "nan_rate": self.nan_count / max(self.total_batches, 1),
            "inf_rate": self.inf_count / max(self.total_batches, 1),
            "skip_rate": self.skipped_batches / max(self.total_batches, 1),
            "ema_grad_norm": self.ema_grad_norm,
            "ema_loss": self.ema_loss,
        }