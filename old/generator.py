"""Behavioral Diffusion Generator — full model assembly.

Ties together:
  - BaseBehaviorEncoder    (market state → B0)
  - AgentBehaviorModule     (B0 + z → per-path Bi)
  - DiffusionUNet           (noise prediction with FiLM conditioning)
  - DDIMScheduler          (iterative reverse diffusion)

Training:  forward()  → predicts noise added to clean returns
Inference: generate() → runs full DDIM sampling loop
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .config import BehaviorGenConfig
from .behavioral_encoder import (
    BaseBehaviorEncoder,
    AgentBehaviorModule,
    SinusoidalTimestepEmbedding,
    _make_timestep_mlp,
)
from .diffusion_unet import DiffusionUNet
from .diffusion_sampler import DDIMScheduler


class BehaviorDiffusionGenerator(nn.Module):
    """Full behavioral diffusion manifold generator.

    Training flow:
      1. Encode B0 from context
      2. Generate per-path Bi from B0 + z
      3. Add noise to clean returns
      4. UNet predicts noise
      5. Compute reconstruction + structural + diversity losses

    Inference flow:
      1. Encode B0, Bi
      2. DDIM sample: noise → 50 steps → clean returns
      3. Return (paths, behaviors, base_behavior)
    """

    def __init__(self, config: Optional[BehaviorGenConfig] = None):
        super().__init__()
        self.config = config or BehaviorGenConfig()

        self.base_encoder = BaseBehaviorEncoder(self.config)
        self.agent_module = AgentBehaviorModule(self.config)
        self.unet = DiffusionUNet(self.config)
        self.scheduler = DDIMScheduler(self.config)

    def forward(
        self,
        noisy_returns: Tensor,       # (B_total, 1, T)
        timestep: Tensor,            # (B_total,)
        behavior: Tensor,            # (B_total, base_behavior_dim)
    ) -> Tensor:
        """Predict noise ε given noisy returns, timestep, and behavior.

        Returns (B_total, 1, T) predicted noise.
        """
        return self.unet(noisy_returns, timestep, behavior)

    def forward_with_context(
        self,
        short_seq: Tensor,           # (B, 120, feature_dim)
        mid_seq: Tensor,             # (B, 240, feature_dim)
        long_seq: Tensor,           # (B, 480, feature_dim)
        clean_returns: Tensor,       # (B, K, T)  or  (B, T)
        num_paths: Optional[int] = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Full training forward pass: encode → noise → predict.

        Returns:
            (noise_pred, noise, noisy_returns, B_agent, B0)
        """
        B = clean_returns.shape[0]

        # Encode
        B0 = self.base_encoder(short_seq, mid_seq, long_seq)           # (B, behavior_dim)

        if clean_returns.ndim == 2:
            # Expand: (B, T) → (B, K, T)
            K = num_paths or self.config.training_paths_per_sample
            clean_returns = clean_returns.unsqueeze(1).expand(-1, K, -1)
        else:
            K = clean_returns.shape[1]

        B_agent, z = self.agent_module(B0, K)                          # (B, K, behavior_dim)

        # Flatten for UNet
        clean_flat = clean_returns.reshape(B * K, 1, -1)               # (B*K, 1, T)
        beh_flat = B_agent.reshape(B * K, -1)                          # (B*K, behavior_dim)

        # Sample timesteps and add noise
        device = clean_flat.device
        t = torch.randint(
            0, self.config.diffusion_timesteps,
            (B * K,), device=device, dtype=torch.long
        )

        noisy, noise = self.scheduler.add_noise(clean_flat, t)

        # Predict noise
        noise_pred = self.forward(noisy, t, beh_flat)

        return noise_pred, noise, noisy, B_agent, B0

    @torch.no_grad()
    def generate(
        self,
        short_seq: Tensor,
        mid_seq: Tensor,
        long_seq: Tensor,
        num_paths: int = 128,
        clip_denoised: bool = True,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Generate future return paths via DDIM sampling.

        Args:
            short_seq:  (B, 120, feature_dim)
            mid_seq:    (B, 240, feature_dim)
            long_seq:   (B, 480, feature_dim)
            num_paths:  Number of generated futures per sample

        Returns:
            (paths, behaviors, base_behavior)
             paths:      (B, num_paths, horizon)  in returns space
             behaviors:  (B, num_paths, behavior_dim)
             base_behavior: (B, behavior_dim)
        """
        return self.scheduler.sample(
            self, short_seq, mid_seq, long_seq,
            num_paths=num_paths,
            clip_denoised=clip_denoised,
        )

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)