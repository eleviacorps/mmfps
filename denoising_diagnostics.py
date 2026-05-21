"""Comprehensive denoising convergence diagnostics for MMFPS_GEN_V2.

Tracks whether the diffusion reverse process is actually refining paths:
  - x0 prediction quality (MSE to target)
  - x0 magnitude calibration (does predicted x0 match target scale?)
  - Noise prediction progression (does MSE improve as t→0?)
  - Signal-to-noise ratio tracking
  - Per-timestep bin analysis
  - Convergence metrics for DDIM step validation
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from config import BehaviorGenConfig


def compute_denoising_metrics(
    x0_pred: Tensor,
    noise_pred: Tensor,
    noise_true: Tensor,
    target_returns: Tensor,
    noisy_returns: Tensor,
    timesteps: Tensor,
    scheduler,
    config: BehaviorGenConfig,
) -> dict[str, float]:
    """Comprehensive denoising quality assessment.

    Args:
        x0_pred: (B, K, T) predicted clean returns
        noise_pred: (B*K, 1, T) predicted noise
        noise_true: (B*K, 1, T) true added noise
        target_returns: (B, T) ground truth returns
        noisy_returns: (B*K, 1, T) noisy input
        timesteps: (B*K,) diffusion timesteps
        scheduler: DDIMScheduler instance
        config: BehaviorGenConfig

    Returns:
        dict with keys like "denoising/x0_mse_to_target", etc.
    """
    metrics = {}
    B, K, T = x0_pred.shape
    B_total = B * K

    # ─── 1. Reconstruction fidelity ───────────────────────────────────

    # Expand target for comparison
    if target_returns.ndim == 2:
        target_exp = target_returns.unsqueeze(1).expand(B, K, T)
    else:
        target_exp = target_returns

    # MSE of predicted x0 vs. true target
    mse_to_target = F.mse_loss(x0_pred, target_exp, reduction='none')  # (B, K, T)
    metrics["denoising/x0_mse_to_target"] = float(mse_to_target.mean().item())
    metrics["denoising/x0_mse_p50"] = float(torch.quantile(mse_to_target, 0.5).item())
    metrics["denoising/x0_mse_p90"] = float(torch.quantile(mse_to_target, 0.9).item())

    # ─── 2. Magnitude calibration (is x0 scale correct?) ────────────

    # At early timesteps (high noise), x0 should be small (mostly noise)
    # At late timesteps (low noise), x0 should be large (close to actual signal)
    alpha_t = scheduler.alphas_cumprod.to(x0_pred.device)[timesteps]  # (B_total,)
    alpha_t_2d = alpha_t.reshape(B, K, 1).expand(B, K, T)

    # Expected signal strength at this timestep
    expected_signal_strength = torch.sqrt(alpha_t_2d)  # Signal component magnitude
    actual_signal_strength = x0_pred.abs()

    # Ratio: should be ~1 if x0 is well-calibrated
    strength_ratio = actual_signal_strength.mean() / (expected_signal_strength.mean() + 1e-6)
    metrics["denoising/x0_strength_ratio"] = float(strength_ratio.item())

    # ─── 3. Noise prediction quality per timestep bin ────────────────

    noise_pred_flat = noise_pred.reshape(B_total, T)
    noise_true_flat = noise_true.reshape(B_total, T)
    noise_mse_per_sample = (noise_pred_flat - noise_true_flat).pow(2).mean(dim=1)  # (B_total,)

    metrics["denoising/noise_mse_mean"] = float(noise_mse_per_sample.mean().item())
    metrics["denoising/noise_mse_std"] = float(noise_mse_per_sample.std().item())
    metrics["denoising/noise_mse_p90"] = float(torch.quantile(noise_mse_per_sample, 0.9).item())

    # ─── 4. Per-timestep bin analysis ─────────────────────────────

    t_int = timesteps.to(torch.int32)
    t_bin_size = max(1, config.diffusion_timesteps // 5)

    for bin_i in range(5):
        lo = bin_i * t_bin_size
        hi = (bin_i + 1) * t_bin_size
        bin_mask = (t_int >= lo) & (t_int < hi)

        if bin_mask.any():
            # Noise prediction error in this bin
            noise_mse_bin = (noise_pred_flat[bin_mask] - noise_true_flat[bin_mask]).pow(2).mean()
            metrics[f"denoising/noise_mse_t{bin_i}"] = float(noise_mse_bin.item())

            # x0 reconstruction error in this bin
            x0_pred_bin = x0_pred.reshape(B_total, T)[bin_mask]
            target_exp_flat = target_exp.reshape(B_total, T)[bin_mask]
            x0_mse_bin = (x0_pred_bin - target_exp_flat).pow(2).mean()
            metrics[f"denoising/x0_mse_t{bin_i}"] = float(x0_mse_bin.item())

            # Average timestep in this bin
            t_avg = timesteps[bin_mask].float().mean()
            metrics[f"denoising/t_avg_t{bin_i}"] = float(t_avg.item())

    # ─── 5. Signal-to-Noise Ratio (SNR) ───────────────────────────

    # SNR = alpha_t / (1 - alpha_t)
    snr = alpha_t / (1.0 - alpha_t + 1e-8)
    metrics["denoising/snr_mean"] = float(snr.mean().item())
    metrics["denoising/snr_std"] = float(snr.std().item())
    metrics["denoising/snr_min"] = float(snr.min().item())
    metrics["denoising/snr_max"] = float(snr.max().item())

    # ─── 6. x0 prediction variance (path diversity) ────────────────

    x0_std_per_timestep = x0_pred.std(dim=(1, 2))  # (B,) — across all paths and time
    metrics["denoising/x0_std_global"] = float(x0_std_per_timestep.mean().item())
    metrics["denoising/x0_std_per_sample_mean"] = float(x0_std_per_timestep.mean().item())
    metrics["denoising/x0_std_per_sample_min"] = float(x0_std_per_timestep.min().item())
    metrics["denoising/x0_std_per_sample_max"] = float(x0_std_per_timestep.max().item())

    # ─── 7. Convergence toward target (directional progress) ───────

    # For each sample, check if x0_pred is moving toward target
    # Metric: cosine similarity between (x0_pred - noisy) and (target - noisy)
    noisy_returns_orig = noisy_returns.reshape(B, K, T)

    # Center around noisy baseline
    x0_centered = x0_pred - noisy_returns_orig
    target_centered = target_exp - noisy_returns_orig

    # Cosine similarity (should be positive if moving toward target)
    dot_product = (x0_centered * target_centered).sum(dim=-1).mean()
    x0_norm = x0_centered.norm(dim=-1).mean()
    target_norm = target_centered.norm(dim=-1).mean()
    cosine_sim = dot_product / (x0_norm * target_norm + 1e-8)

    metrics["denoising/cosine_similarity_to_target"] = float(cosine_sim.item())

    # ─── 8. Gradient statistics (without explicit backprop) ────────

    # Numerical gradient estimate via finite differences
    # ∂x0/∂noise ≈ (x0_pred - target) / sqrt(1 - alpha_t)
    noise_component_scale = torch.sqrt(1.0 - alpha_t_2d + 1e-8)
    implicit_grad = (x0_pred - target_exp) / (noise_component_scale + 1e-8)
    implicit_grad_norm = implicit_grad.norm(dim=-1).mean()

    metrics["denoising/implicit_grad_norm"] = float(implicit_grad_norm.item())

    # ─── 9. Noise prediction improvement trend ─────────────────────

    # Bucket by timestep and track whether noise MSE improves at later stages
    early_bins = []
    late_bins = []

    for bin_i in range(5):
        lo = bin_i * t_bin_size
        hi = (bin_i + 1) * t_bin_size
        bin_mask = (t_int >= lo) & (t_int < hi)

        if bin_mask.any():
            noise_mse_bin = (noise_pred_flat[bin_mask] - noise_true_flat[bin_mask]).pow(2).mean()

            if bin_i < 2:
                early_bins.append(noise_mse_bin.item())
            else:
                late_bins.append(noise_mse_bin.item())

    if early_bins and late_bins:
        early_avg = sum(early_bins) / len(early_bins)
        late_avg = sum(late_bins) / len(late_bins)
        improvement = (early_avg - late_avg) / (early_avg + 1e-8)
        metrics["denoising/noise_mse_improvement"] = float(improvement)
    else:
        metrics["denoising/noise_mse_improvement"] = 0.0

    # ─── 10. Finite value checking ─────────────────────────────────

    metrics["denoising/x0_has_nan"] = float(torch.isnan(x0_pred).any().item())
    metrics["denoising/x0_has_inf"] = float(torch.isinf(x0_pred).any().item())
    metrics["denoising/noise_has_nan"] = float(torch.isnan(noise_pred).any().item())

    return metrics


def diagnose_ddim_step_quality(
    x_t_before: Tensor,
    x_t_after: Tensor,
    eps_pred: Tensor,
    t: int,
    target_distribution: Tensor | None = None,
) -> dict[str, float]:
    """Assess quality of a single DDIM reverse step.

    Args:
        x_t_before: (B, K, T) before DDIM step
        x_t_after: (B, K, T) after DDIM step
        eps_pred: (B, K, T) predicted noise
        t: current timestep
        target_distribution: (B, K, T) reference clean data

    Returns:
        dict with DDIM step diagnostics
    """
    metrics = {}

    # Change in magnitude
    delta = (x_t_after - x_t_before).abs().mean()
    metrics["ddim_step/change_magnitude"] = float(delta.item())

    # Noise amplitude
    eps_magnitude = eps_pred.abs().mean()
    metrics["ddim_step/eps_magnitude"] = float(eps_magnitude.item())

    # Direction of change (is it reducing noise?)
    # Should be moving in direction of lower noise variance
    x_after_var = x_t_after.var()
    x_before_var = x_t_before.var()
    variance_reduction = (x_before_var - x_after_var) / (x_before_var + 1e-8)
    metrics["ddim_step/variance_reduction"] = float(variance_reduction.item())

    if target_distribution is not None:
        # Distance to clean data
        dist_before = (x_t_before - target_distribution).pow(2).mean()
        dist_after = (x_t_after - target_distribution).pow(2).mean()
        metrics["ddim_step/dist_to_target_before"] = float(dist_before.item())
        metrics["ddim_step/dist_to_target_after"] = float(dist_after.item())
        metrics["ddim_step/improvement_to_target"] = float((dist_before - dist_after).item())

    return metrics
