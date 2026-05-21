"""Pure diffusion reconstruction loss.

Phase 1: train only noise-prediction MSE until denoiser converges.
Structural/financial losses will be re-introduced in Phase 2.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def reconstruction_loss(noise_pred: Tensor, noise_true: Tensor) -> Tensor:
    """Standard diffusion noise-prediction MSE.

    This is THE training signal. Everything else is diagnostic.
    """
    return F.mse_loss(noise_pred, noise_true)


def compute_all_losses(
    noise_pred: Tensor,
    noise_true: Tensor,
    predicted_returns: Tensor,
    target_returns: Tensor,
    behaviors: Tensor,
    config,
    alpha_t_mean: float = 1.0,
) -> dict[str, Tensor]:
    """Reconstruction only. Structural losses are computed as detached
    diagnostics for logging but do NOT contribute gradients.

    Args:
        noise_pred: (B*K, 1, T) predicted noise
        noise_true: (B*K, 1, T) true noise
        predicted_returns: (B, K, T) x0 prediction (diagnostic only)
        target_returns: (B, T) ground truth (diagnostic only)
        behaviors: (B, K, D) behavior embeddings (diagnostic only)
        config: unused in Phase 1 (structural losses disabled)
        alpha_t_mean: unused in Phase 1
    """
    losses = {}
    losses["reconstruction"] = reconstruction_loss(noise_pred, noise_true)

    with torch.no_grad():
        target_exp = target_returns.unsqueeze(1).expand(
            -1, predicted_returns.shape[1], -1
        ) if target_returns.ndim == 2 else target_returns

        losses["diag/volatility"] = F.l1_loss(
            _volatility_profile(predicted_returns),
            _volatility_profile(target_exp),
        )
        pred_cumsum = predicted_returns.cumsum(dim=-1)
        tgt_cumsum = target_exp.cumsum(dim=-1)
        losses["diag/trend"] = F.mse_loss(pred_cumsum, tgt_cumsum) / predicted_returns.shape[-1]
        losses["diag/volatility"].detach_()
        losses["diag/trend"].detach_()

    losses["total"] = losses["reconstruction"]

    return losses


def _volatility_profile(x: Tensor, windows: list[int] | None = None) -> Tensor:
    """Multi-scale rolling volatility. Detached diagnostic only."""
    if windows is None:
        windows = [2, 5, 10, 20]
    B, K, T = x.shape
    profile = x.new_zeros(B, K, len(windows))
    for wi, w in enumerate(windows):
        if T < w:
            profile[:, :, wi] = x.std(dim=-1)
        else:
            unfolded = x.unfold(-1, w, 1)
            profile[:, :, wi] = unfolded.std(dim=-1).mean(dim=-1)
    return profile
