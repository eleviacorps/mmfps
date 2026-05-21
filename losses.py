"""Diffusion losses for reconstruction and path-level financial structure.

Stage A can stay pure reconstruction by setting all structural weights to 0.
Later stages add best-of-K path matching, volatility/trend structure, temporal
smoothness, and diversity pressure so the generated bundle is rewarded for
containing a realistic future instead of averaging toward one path.
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
    """Compute diffusion reconstruction plus optional structural losses.

    Args:
        noise_pred: (B*K, 1, T) predicted noise
        noise_true: (B*K, 1, T) true noise
        predicted_returns: (B, K, T) x0 prediction in scaled return space
        target_returns: (B, T) ground truth in scaled return space
        behaviors: (B, K, D) per-path behavior embeddings
        config: loss weights and diversity margins
        alpha_t_mean: unused in Phase 1
    """
    losses = {}
    losses["reconstruction"] = reconstruction_loss(noise_pred, noise_true)

    target_exp = target_returns.unsqueeze(1).expand(
        -1, predicted_returns.shape[1], -1
    ) if target_returns.ndim == 2 else target_returns

    best_idx, best_paths = _best_matching_paths(predicted_returns, target_returns)

    losses["volatility"] = F.l1_loss(
        _volatility_profile(best_paths.unsqueeze(1)),
        _volatility_profile(target_returns.unsqueeze(1)),
    )
    losses["trend"] = _trend_loss(best_paths, target_returns)
    losses["turning"] = _directional_continuity_loss(best_paths, target_returns)
    losses["temporal_smoothness"] = _smoothness_loss(best_paths, target_returns)
    losses["autocorr"] = _autocorr_loss(best_paths, target_returns)
    losses["tail"] = _tail_loss(best_paths, target_returns)
    losses["diversity"] = _diversity_loss(predicted_returns, config)
    losses["manifold_spread"] = _manifold_spread_loss(predicted_returns, config)
    losses["latent_sensitivity"] = _latent_sensitivity_loss(predicted_returns, behaviors, config)

    losses["total"] = (
        config.weight_mse * losses["reconstruction"]
        + config.weight_volatility * losses["volatility"]
        + config.weight_trend * losses["trend"]
        + config.weight_turning * losses["turning"]
        + getattr(config, "weight_smoothness", 0.0) * losses["temporal_smoothness"]
        + getattr(config, "weight_autocorr", 0.0) * losses["autocorr"]
        + getattr(config, "weight_tail", 0.0) * losses["tail"]
        + config.weight_diversity * losses["diversity"]
        + config.weight_manifold_spread * losses["manifold_spread"]
        + config.weight_latent_sensitivity * losses["latent_sensitivity"]
    )

    with torch.no_grad():
        losses["diag/best_path_index_mean"] = best_idx.float().mean()
        losses["diag/best_path_mse"] = ((best_paths - target_returns) ** 2).mean()

    return losses


def _best_matching_paths(paths: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    """Pick the path that best matches direction, magnitude, volatility, and structure."""
    target_exp = target.unsqueeze(1)
    point_mse = ((paths - target_exp) ** 2).mean(dim=-1)
    magnitude = (paths[:, :, -1] - target[:, None, -1]).abs()
    volatility = (paths.std(dim=-1) - target.std(dim=-1, keepdim=True)).abs()
    structure = ((paths.cumsum(dim=-1) - target_exp.cumsum(dim=-1)) ** 2).mean(dim=-1)
    path_dir = torch.sign(paths[:, :, -1] - paths[:, :, 0])
    target_dir = torch.sign(target[:, -1] - target[:, 0]).unsqueeze(1)
    direction_penalty = (path_dir != target_dir).float()
    score = point_mse + 0.25 * magnitude + 0.25 * volatility + 0.25 * structure + 0.05 * direction_penalty
    best_idx = score.argmin(dim=1)
    best = paths[torch.arange(paths.shape[0], device=paths.device), best_idx]
    return best_idx, best


def _trend_loss(path: Tensor, target: Tensor) -> Tensor:
    pred_cumsum = path.cumsum(dim=-1)
    tgt_cumsum = target.cumsum(dim=-1)
    endpoint = F.smooth_l1_loss(pred_cumsum[:, -1], tgt_cumsum[:, -1])
    shape = F.mse_loss(pred_cumsum, tgt_cumsum) / path.shape[-1]
    return shape + 0.5 * endpoint


def _directional_continuity_loss(path: Tensor, target: Tensor) -> Tensor:
    pred_steps = path[:, 1:] - path[:, :-1]
    tgt_steps = target[:, 1:] - target[:, :-1]
    local = F.l1_loss(torch.tanh(pred_steps), torch.tanh(tgt_steps))
    pred_dir = torch.tanh(path.sum(dim=-1))
    tgt_dir = torch.tanh(target.sum(dim=-1))
    global_dir = F.mse_loss(pred_dir, tgt_dir)
    return local + global_dir


def _smoothness_loss(path: Tensor, target: Tensor) -> Tensor:
    if path.shape[-1] < 3:
        return path.new_zeros(())
    pred_curv = path[:, 2:] - 2.0 * path[:, 1:-1] + path[:, :-2]
    tgt_curv = target[:, 2:] - 2.0 * target[:, 1:-1] + target[:, :-2]
    return F.smooth_l1_loss(pred_curv, tgt_curv)


def _autocorr_loss(path: Tensor, target: Tensor) -> Tensor:
    return F.l1_loss(_lag_autocorr(path, lag=1), _lag_autocorr(target, lag=1))


def _tail_loss(path: Tensor, target: Tensor) -> Tensor:
    pred = _standardized_moment(path, order=4)
    tgt = _standardized_moment(target, order=4)
    return F.smooth_l1_loss(pred, tgt)


def _diversity_loss(paths: Tensor, config) -> Tensor:
    if paths.shape[1] < 2:
        return paths.new_zeros(())
    dists = torch.cdist(paths, paths, p=2) / (paths.shape[-1] ** 0.5)
    tri = torch.triu(torch.ones_like(dists, dtype=torch.bool), diagonal=1)
    pairwise = dists[tri].view(paths.shape[0], -1)
    mean_dist = pairwise.mean(dim=1)
    return F.relu(config.diversity_min_distance - mean_dist).mean()


def _manifold_spread_loss(paths: Tensor, config) -> Tensor:
    endpoint_spread = paths[:, :, -1].std(dim=1)
    vol_spread = paths.std(dim=-1).std(dim=1)
    spread = 0.7 * endpoint_spread + 0.3 * vol_spread
    return F.relu(config.manifold_min_spread - spread).mean()


def _latent_sensitivity_loss(paths: Tensor, behaviors: Tensor, config) -> Tensor:
    if paths.shape[1] < 2:
        return paths.new_zeros(())
    path_dist = torch.cdist(paths, paths, p=2) / (paths.shape[-1] ** 0.5)
    beh_dist = torch.cdist(behaviors, behaviors, p=2) / (behaviors.shape[-1] ** 0.5)
    tri = torch.triu(torch.ones_like(path_dist, dtype=torch.bool), diagonal=1)
    ratio = path_dist[tri] / (beh_dist[tri].detach() + 1e-4)
    return F.relu(config.latent_sensitivity_min_distance - ratio).mean()


def _lag_autocorr(x: Tensor, lag: int = 1) -> Tensor:
    if x.shape[-1] <= lag:
        return x.new_zeros(x.shape[0])
    a = x[:, :-lag] - x[:, :-lag].mean(dim=-1, keepdim=True)
    b = x[:, lag:] - x[:, lag:].mean(dim=-1, keepdim=True)
    denom = a.square().mean(dim=-1).sqrt() * b.square().mean(dim=-1).sqrt()
    return (a * b).mean(dim=-1) / (denom + 1e-6)


def _standardized_moment(x: Tensor, order: int) -> Tensor:
    centered = x - x.mean(dim=-1, keepdim=True)
    std = centered.std(dim=-1, keepdim=True).clamp_min(1e-4)
    return (centered / std).pow(order).mean(dim=-1)


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
