"""Loss functions for MMFPS_GEN_V2 behavioral diffusion training.

All losses operate in returns space. The generator minimizes a weighted sum
of reconstruction, structural, diversity, and manifold-spread terms.

Design principle:
  - Reconstruction: get close to the true return path (MSE)
  - Structural:  match volatility shape, trend direction, turning points
  - Diversity:  paths should NOT all be the same (anti-collapse)
  - Manifold:   endpoint spread ensures wide coverage
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from .config import BehaviorGenConfig


# ── Helper: windowed volatility profile ────────────────────────────────────

@torch.no_grad()
def _volatility_profile(x: Tensor, windows: list[int] = [2, 5, 10, 20]) -> Tensor:
    """Compute rolling std for each window size → concatenate.

    Args:
        x: (B, K, T)
        windows: list of window sizes
    Returns:
        profile: (B, K, len(windows))
    """
    B, K, T = x.shape
    profile = torch.zeros(B, K, len(windows), device=x.device, dtype=x.dtype)

    for wi, w in enumerate(windows):
        if T < w:
            profile[:, :, wi] = x.std(dim=-1)
        else:
            # Rolling std via unfold → std over window dim
            unfolded = x.unfold(-1, w, 1)                      # (B, K, T-w+1, w)
            profile[:, :, wi] = unfolded.std(dim=-1).mean(dim=-1)

    return profile


# ── Reconstruction ──────────────────────────────────────────────────────────

def reconstruction_loss(noise_pred: Tensor, noise_true: Tensor) -> Tensor:
    """Standard diffusion noise-prediction MSE.

    This is the primary training signal: how well the UNet predicts
    the noise that was added to the clean data.
    """
    return F.mse_loss(noise_pred, noise_true)


# ── Structural losses ───────────────────────────────────────────────────────

def volatility_loss(pred_returns: Tensor, target_returns: Tensor) -> Tensor:
    """Match multi-scale volatility profiles.

    Args:
        pred_returns:   (B, K, T) generated returns
        target_returns: (B, T) or (B, K, T) real returns
    """
    if target_returns.ndim == 2:
        target_returns = target_returns.unsqueeze(1).expand(-1, pred_returns.shape[1], -1)

    pred_profile = _volatility_profile(pred_returns)
    tgt_profile = _volatility_profile(target_returns)

    return F.l1_loss(pred_profile, tgt_profile)


def trend_loss(pred_returns: Tensor, target_returns: Tensor) -> Tensor:
    """Match cumulative return trajectory (directional consistency).

    Normalized by T so magnitude stays comparable to reconstruction loss.
    """
    T = pred_returns.shape[-1]
    if target_returns.ndim == 2:
        target_returns = target_returns.unsqueeze(1).expand(-1, pred_returns.shape[1], -1)

    pred_cumsum = pred_returns.cumsum(dim=-1)
    tgt_cumsum = target_returns.cumsum(dim=-1)

    raw_mse = F.mse_loss(pred_cumsum, tgt_cumsum)
    return raw_mse / T


def turning_loss(pred_returns: Tensor, target_returns: Tensor) -> Tensor:
    """Penalize mismatch in sign-change frequency (turning points)."""
    if target_returns.ndim == 2:
        target_returns = target_returns.unsqueeze(1).expand(-1, pred_returns.shape[1], -1)

    # Count sign changes (turning points)
    def _turning_count(x: Tensor) -> Tensor:
        signs = torch.sign(x)
        # Sign change:  diff of sign != 0
        changes = (signs[..., 1:] - signs[..., :-1]).abs().clamp(0, 1)
        return changes.sum(dim=-1).float() / (x.shape[-1] - 1)  # Normalize

    pred_turns = _turning_count(pred_returns)         # (B, K)
    tgt_turns = _turning_count(target_returns)         # (B,) or (B, K)

    if tgt_turns.ndim == 1:
        tgt_turns = tgt_turns.unsqueeze(-1).expand(-1, pred_returns.shape[1])

    return F.mse_loss(pred_turns, tgt_turns)


# ── Diversity pressure ──────────────────────────────────────────────────────

def diversity_loss(paths: Tensor, config: BehaviorGenConfig | None = None) -> Tensor:
    """Pairwise path repulsion: maintain minimum mean pairwise distance.

    Uses margin-based formulation: penalty only if mean distance < min_distance.
    Replaces unbounded log-based formulation to prevent gradient explosion.

    Args:
        paths: (B, K, T) generated returns
        config: BehaviorGenConfig (for min_distance threshold)

    Returns:
        loss: scalar, 0 if mean_dist >= min_distance, else (min_distance - mean_dist)^2
    """
    B, K, T = paths.shape
    paths_2d = paths.reshape(B * K, T)                  # (B*K, T)

    pairwise_dists = torch.cdist(paths_2d, paths_2d, p=2)
    mask = (~torch.eye(B * K, device=paths.device).bool())
    mean_dist = pairwise_dists[mask].mean()

    min_distance = 0.25 if config is None else config.diversity_min_distance
    
    # Margin penalty: only penalize if below threshold
    deficit = torch.relu(min_distance - mean_dist)
    return deficit ** 2


def latent_diversity_loss(behaviors: Tensor) -> Tensor:
    """Encourage behavior embeddings to be diverse via eigenvalue maximization.

    Uses eigenvalue-based formulation instead of log-determinant to avoid
    numerical instability and unbounded gradients near singular matrices.

    Args:
        behaviors: (B, K, D) per-path behavior embeddings

    Returns:
        loss: scalar, negative of mean eigenvalue (lower = more diverse)
    """
    B, K, D = behaviors.shape
    beh_flat = behaviors.reshape(-1, D)                   # (B*K, D)

    # Center
    beh_centered = beh_flat - beh_flat.mean(dim=0, keepdim=True)

    # Covariance
    cov = (beh_centered.T @ beh_centered) / max(beh_flat.shape[0] - 1, 1)  # (D, D)

    # Add small identity for numerical stability
    eps = 1e-3
    cov_reg = cov + eps * torch.eye(D, device=cov.device, dtype=cov.dtype)

    # Eigendecomposition (more stable than logdet)
    try:
        eigvals = torch.linalg.eigvalsh(cov_reg)
    except RuntimeError:
        # Fallback: compute via SVD
        _, eigvals, _ = torch.svd(cov_reg)

    # Clamp eigenvalues to avoid numerical issues
    eigvals = torch.clamp(eigvals, min=1e-6)

    # Maximize mean eigenvalue (minimize negative mean)
    # Larger eigenvalues = more spread in latent space
    return -eigvals.mean() / D


def latent_sensitivity_loss(paths: Tensor, config: BehaviorGenConfig | None = None) -> Tensor:
    """Ensure latent variation translates to output variation.

    Uses margin-based formulation instead of unbounded log penalty.
    Penalizes when different latents produce nearly identical paths.

    Args:
        paths: (B, K, T) generated returns
        config: BehaviorGenConfig (for min_distance threshold)

    Returns:
        loss: scalar, margin-based penalty
    """
    B, K, T = paths.shape
    paths_2d = paths.reshape(B * K, T)

    pairwise_dists = torch.cdist(paths_2d, paths_2d, p=2)
    mask = (~torch.eye(B * K, device=paths.device).bool())
    mean_dist = pairwise_dists[mask].mean()

    min_distance = 0.15 if config is None else config.latent_sensitivity_min_distance
    
    # Margin penalty: enforce minimum latent sensitivity
    deficit = torch.relu(min_distance - mean_dist)
    return deficit ** 2


# ── Manifold spread ─────────────────────────────────────────────────────────

def manifold_spread_loss(paths: Tensor, config: BehaviorGenConfig | None = None) -> Tensor:
    """Anti-collapse: encourage wide endpoint distribution.

    Uses margin-based formulation instead of unbounded log penalty.
    Reward high standard deviation of final-step values across paths.
    This directly prevents all paths ending at the same price.

    Args:
        paths: (B, K, T) generated returns
        config: BehaviorGenConfig (for min_spread threshold)

    Returns:
        loss: scalar, 0 if spread >= min_spread, else (min_spread - spread)^2
    """
    endpoints = paths[:, :, -1]                         # (B, K)
    spread = endpoints.std(dim=1).mean()                # Mean of per-sample spread

    min_spread = 0.20 if config is None else config.manifold_min_spread

    # Margin penalty: only penalize if below threshold
    deficit = torch.relu(min_spread - spread)
    return deficit ** 2


# ── Combined loss computation ───────────────────────────────────────────────

def compute_all_losses(
    noise_pred: Tensor,
    noise_true: Tensor,
    predicted_returns: Tensor,
    target_returns: Tensor,
    behaviors: Tensor,
    config: BehaviorGenConfig,
) -> dict[str, Tensor]:
    """Compute all training losses and return as a dict (for logging).
    
    Also computes per-component diagnostics to detect which loss is destabilizing.
    """
    w = config

    losses = {}
    losses["reconstruction"] = reconstruction_loss(noise_pred, noise_true)
    losses["volatility"] = volatility_loss(predicted_returns, target_returns)

    target_exp = target_returns.unsqueeze(1).expand(-1, predicted_returns.shape[1], -1) if target_returns.ndim == 2 else target_returns
    pred_cumsum = predicted_returns.cumsum(dim=-1)
    tgt_cumsum = target_exp.cumsum(dim=-1)
    raw_trend = F.mse_loss(pred_cumsum, tgt_cumsum)
    losses["trend"] = raw_trend / predicted_returns.shape[-1]
    losses["trend_raw"] = raw_trend.detach()

    losses["turning"] = turning_loss(predicted_returns, target_returns)
    losses["diversity"] = diversity_loss(predicted_returns, config)
    losses["latent_diversity"] = latent_diversity_loss(behaviors)
    losses["latent_sensitivity"] = latent_sensitivity_loss(predicted_returns, config)
    losses["manifold_spread"] = manifold_spread_loss(predicted_returns, config)

    # ─── Compute weighted contributions ───────────────────────────────
    weighted_losses = {
        "reconstruction": w.weight_mse * losses["reconstruction"],
        "volatility": w.weight_volatility * losses["volatility"],
        "trend": w.weight_trend * losses["trend"],
        "turning": w.weight_turning * losses["turning"],
        "diversity": w.weight_diversity * losses["diversity"],
        "latent_sensitivity": w.weight_latent_sensitivity * losses["latent_sensitivity"],
        "manifold_spread": w.weight_manifold_spread * losses["manifold_spread"],
    }

    # ─── Diagnostic: magnitude of each loss component ─────────────────
    for name, loss_val in weighted_losses.items():
        losses[f"{name}_weighted"] = loss_val

    total = sum(weighted_losses.values())
    losses["total"] = total

    return losses