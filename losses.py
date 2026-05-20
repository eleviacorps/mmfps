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

def diversity_loss(paths: Tensor) -> Tensor:
    """Pairwise path repulsion: maximize mean pairwise distance.

    Uses negative log of mean pairwise distance — paths that are too
    similar receive high penalty.
    """
    B, K, T = paths.shape
    paths_2d = paths.reshape(B * K, T)                  # (B*K, T)

    pairwise_dists = torch.cdist(paths_2d, paths_2d, p=2)
    mask = (~torch.eye(B * K, device=paths.device).bool())
    mean_dist = pairwise_dists[mask].mean()

    return -torch.log(mean_dist + 1e-6)


def latent_diversity_loss(behaviors: Tensor) -> Tensor:
    """Encourage behavior embeddings to be diverse (covariance maximization).

    Maximizes the log-determinant of the behavior covariance matrix,
    which pushes embeddings to span the available latent space.
    """
    B, K, D = behaviors.shape
    beh_flat = behaviors.reshape(-1, D)                   # (B*K, D)

    # Center
    beh_centered = beh_flat - beh_flat.mean(dim=0, keepdim=True)

    # Covariance
    cov = (beh_centered.T @ beh_centered) / (beh_flat.shape[0] - 1)  # (D, D)

    # Log-determinant: max(det) → min(-log(det))
    # Add small identity for numerical stability
    eps = 1e-3
    cov_reg = cov + eps * torch.eye(D, device=cov.device)

    try:
        log_det = torch.logdet(cov_reg)
    except RuntimeError:
        # Fallback: eigendecomposition
        eigvals = torch.linalg.eigvalsh(cov_reg)
        log_det = torch.log(eigvals.clamp(min=1e-6)).sum()

    # We minimize, so negate the log-det → larger determinant = lower loss
    return -log_det / D  # Normalize by dimension


def latent_sensitivity_loss(paths: Tensor) -> Tensor:
    """Ensure latent variation translates to output variation.

    Penalizes when different latents produce nearly identical paths.
    """
    B, K, T = paths.shape
    paths_2d = paths.reshape(B * K, T)

    pairwise_dists = torch.cdist(paths_2d, paths_2d, p=2)
    mask = (~torch.eye(B * K, device=paths.device).bool())
    mean_dist = pairwise_dists[mask].mean()

    return -torch.log(mean_dist + 1e-6)


# ── Manifold spread ─────────────────────────────────────────────────────────

def manifold_spread_loss(paths: Tensor) -> Tensor:
    """Anti-collapse: encourage wide endpoint distribution.

    Reward high standard deviation of final-step values across paths.
    This directly prevents all paths ending at the same price.
    """
    # std over all paths and batch
    endpoints = paths[:, :, -1]                    # (B, K)
    spread = endpoints.std(dim=1).mean()           # Mean of per-sample spread

    return -torch.log(spread + 1e-6)


# ── Combined loss computation ───────────────────────────────────────────────

def compute_all_losses(
    noise_pred: Tensor,
    noise_true: Tensor,
    predicted_returns: Tensor,
    target_returns: Tensor,
    behaviors: Tensor,
    config: BehaviorGenConfig,
) -> dict[str, Tensor]:
    """Compute all training losses and return as a dict (for logging)."""
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
    losses["diversity"] = diversity_loss(predicted_returns)
    losses["latent_diversity"] = latent_diversity_loss(behaviors)
    losses["latent_sensitivity"] = latent_sensitivity_loss(predicted_returns)
    losses["manifold_spread"] = manifold_spread_loss(predicted_returns)

    total = (
        w.weight_mse * losses["reconstruction"]
        + w.weight_volatility * losses["volatility"]
        + w.weight_trend * losses["trend"]
        + w.weight_turning * losses["turning"]
        + w.weight_dtw * 0.0
        + w.weight_diversity * losses["diversity"]
        + w.weight_latent_sensitivity * losses["latent_sensitivity"]
        + w.weight_manifold_spread * losses["manifold_spread"]
    )
    losses["total"] = total

    return losses