from __future__ import annotations

"""DDIM scheduler with cosine noise schedule for MMFPS_GEN_V2.

Implements:
  - Forward diffusion:  x_t = √(ᾱ_t)·x_0 + √(1-ᾱ_t)·ε
  - Reverse diffusion:  DDIM step from t to t-1
  - Cosine β schedule (Nichol & Dhariwal, 2021)

The sampler is standalone — it wraps the generator model and performs
iterative denoising from pure noise to structured returns trajectories.
"""

import math
from typing import Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

from .config import BehaviorGenConfig


class DDIMScheduler:
    """Cosine-scheduled DDIM sampler.

    Usage:
        sched = DDIMScheduler(num_timesteps=400)         # quick test
        sched = DDIMScheduler(BehaviorGenConfig())       # full config
        # Training: add noise to clean data
        noisy, noise = sched.add_noise(x0, timesteps)
        # Inference: iterative denoising
        paths = sched.sample(model, context, num_paths=128)
    """

    def __init__(self, config: Union[BehaviorGenConfig, int] = BehaviorGenConfig()):
        if isinstance(config, int):
            self.num_train_timesteps = config
            self.num_inference_steps = max(1, config // 8)  # DDIM default stride
            self.eta = 0.0
            self.noise_scale_val = 1.0
        else:
            self.num_train_timesteps = config.diffusion_timesteps
            self.num_inference_steps = config.sampling_steps
            self.eta = config.sampling_eta
            self.noise_scale_val = config.noise_scale

        # Build β schedule
        betas = self._cosine_beta_schedule(self.num_train_timesteps)
        self.betas = betas
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)

        # Precompute DDIM step coefficients
        self.alphas_cumprod_prev = torch.cat([
            torch.tensor([1.0]),
            self.alphas_cumprod[:-1]
        ])

        self.sqrt_alpha_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    @staticmethod
    def _cosine_beta_schedule(timesteps: int, s: float = 0.008) -> Tensor:
        """Cosine β schedule from "Improved DDPM" (Nichol & Dhariwal, 2021)."""
        steps = timesteps + 1
        t = torch.linspace(0, timesteps, steps, dtype=torch.float64)
        alphas_cumprod = torch.cos((t / timesteps + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, max=0.999).float()

    def add_noise(
        self,
        x0: Tensor,
        timesteps: Tensor,
        noise: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """Forward diffusion: x_t = √(ᾱ_t)·x_0 + √(1-ᾱ_t)·ε

        Args:
            x0: Clean data (B_total, 1, T)
            timesteps: Timesteps (B_total,) in [0, num_train_timesteps)
            noise: Optional pre-sampled noise

        Returns:
            (noisy_x, noise): Noisy sample and the noise added
        """
        if noise is None:
            noise = torch.randn_like(x0)

        # Gather schedule coefficients for requested timesteps
        sqrt_alpha = self.sqrt_alpha_cumprod.to(x0.device)[timesteps]
        sqrt_one_minus = self.sqrt_one_minus_alpha_cumprod.to(x0.device)[timesteps]

        # Reshape for broadcasting: (B,) → (B, 1, 1)
        sqrt_alpha = sqrt_alpha.view(-1, 1, 1)
        sqrt_one_minus = sqrt_one_minus.view(-1, 1, 1)

        noisy = sqrt_alpha * x0 + sqrt_one_minus * noise
        return noisy, noise

    def _get_inference_timesteps(self, device: torch.device) -> Tensor:
        """Sub-sample timesteps for DDIM acceleration.

        DDIM:  use stride = T_train / T_infer
        E.g., 400 / 50 = 8 step stride
        """
        ratio = self.num_train_timesteps // self.num_inference_steps
        steps = torch.arange(
            0, self.num_train_timesteps, ratio,
            device=device, dtype=torch.long
        ).flip(0)  # Reverse order: T, T-ratio, ...
        return steps

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        short_seq: Tensor,
        mid_seq: Tensor,
        long_seq: Tensor,
        num_paths: int = 128,
        clip_denoised: bool = True,
        clip_range: float = 0.5,
    ) -> Tensor:
        """Iterative DDIM reverse sampling.

        Args:
            model: BehaviorDiffusionGenerator (predicts noise from noisy_returns, t, context)
            short_seq:  (B, 120, feature_dim)
            mid_seq:    (B, 240, feature_dim)
            long_seq:   (B, 480, feature_dim)
            num_paths:  Number of future paths to generate per sample
            clip_denoised: Whether to clip predicted x0 to [-clip_range, +clip_range]
            clip_range: Max return magnitude

        Returns:
            paths: (B, num_paths, path_horizon)  in returns space
        """
        B = short_seq.shape[0]
        T = model.config.path_horizon
        device = short_seq.device

        # Encode base behavior and generate per-path behaviors
        B0 = model.base_encoder(short_seq, mid_seq, long_seq)      # (B, behavior_dim)
        B_agent, z = model.agent_module(B0, num_paths)              # (B, K, behavior_dim)

        # Initialize from scaled pure noise
        x_t = self.noise_scale_val * torch.randn(B, num_paths, T, device=device)  # (B, K, T)

        # Get DDIM timestep sequence
        timesteps = self._get_inference_timesteps(device)
        num_steps = len(timesteps)

        for i, t in enumerate(timesteps):
            t_batch = t.unsqueeze(0).expand(B * num_paths).to(device)

            # Flatten paths and behaviors
            x_flat = x_t.reshape(B * num_paths, 1, T)
            beh_flat = B_agent.reshape(B * num_paths, -1)

            # Predict noise
            eps_pred = model.unet(x_flat, t_batch, beh_flat)       # (B*K, 1, T)
            eps_pred = eps_pred.reshape(B, num_paths, T)

            # DDIM step
            x_t = self._ddim_step(x_t, eps_pred, t.item(), i < num_steps - 1)

            if clip_denoised:
                x_t = torch.clamp(x_t, -clip_range, clip_range)

        return x_t, B_agent, B0

    def _ddim_step(
        self,
        x_t: Tensor,
        eps_pred: Tensor,
        t: int,
        is_not_last: bool,
    ) -> Tensor:
        """Single DDIM reverse step.

        Formula:
          x₀_pred = (x_t − √(1−ᾱ_t)·ε_pred) / √(ᾱ_t)
          if last step: return x₀_pred
          x_{prev} = √(ᾱ_{prev})·x₀_pred + √(1−ᾱ_{prev}−σ²_t)·ε_pred + σ_t·ε

        With η=0 (DDIM): σ_t = 0, so x_{prev} is deterministic.
        """
        device = x_t.device

        alpha_t = self.alphas_cumprod[t].to(device)
        alpha_prev = self.alphas_cumprod_prev[t].to(device) if is_not_last else torch.tensor(1.0, device=device)

        # Predict x0
        x0_pred = (x_t - torch.sqrt(1.0 - alpha_t) * eps_pred) / (torch.sqrt(alpha_t) + 1e-8)

        if not is_not_last:
            return x0_pred

        # DDIM stochastic noise (η=0 → deterministic)
        sigma_t = self.eta * torch.sqrt(
            (1.0 - alpha_prev) / (1.0 - alpha_t + 1e-8)
            * (1.0 - alpha_t / (alpha_prev + 1e-8))
        )

        noise = torch.randn_like(x_t)
        x_prev = (
            torch.sqrt(alpha_prev) * x0_pred
            + torch.sqrt(1.0 - alpha_prev - sigma_t**2 + 1e-8) * eps_pred
            + sigma_t * noise
        )

        return x_prev