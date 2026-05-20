"""Multi-horizon behavioral encoder with per-path agent conditioning.

Port of the two-level behavior architecture from MMFPS_V2, cleaned up
and modularized:

  Level 1 — BaseBehaviorEncoder:  3 GRUs (short/mid/long) → fused B0
  Level 2 — AgentBehaviorModule:  B0 + per-path z → diverse Bi
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .config import BehaviorGenConfig


# ══════════════════════════════════════════════════════════════════════════════
# Timestep embedding
# ══════════════════════════════════════════════════════════════════════════════


class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps.

    Uses the standard Transformer-style sinusoids — no learned parameters,
    so it generalizes to any sampling step count.

    Reference: "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
    """

    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: Tensor) -> Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


def _make_timestep_mlp(in_dim: int, out_dim: int) -> nn.Sequential:
    """Standard 2-layer MLP for projecting timestep embeddings."""
    hidden = out_dim * 4
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.SiLU(),
        nn.Linear(hidden, out_dim),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Behavioural Encoder (Level 1)
# ══════════════════════════════════════════════════════════════════════════════


class BaseBehaviorEncoder(nn.Module):
    """Encodes global market state from three temporal horizons.

    Architecture:
        short_encoder : GRU  120→896  (local momentum)
        mid_encoder   : GRU  240→896  (medium regime)
        long_encoder  : GRU  480→896  (macro structure)
          → concat(3×896) → fusion MLP → B0 [896]

    B0 is the "base behavior" — a single market-level interpretation shared
    across all future paths.
    """

    def __init__(self, config: BehaviorGenConfig):
        super().__init__()
        cfg = config
        bh = cfg.base_behavior_dim   # 896

        self.short_encoder = nn.GRU(
            input_size=1,
            hidden_size=bh,
            num_layers=cfg.gru_layers,
            batch_first=True,
            dropout=0.1 if cfg.gru_layers > 1 else 0.0,
        )
        self.mid_encoder = nn.GRU(
            input_size=1,
            hidden_size=bh,
            num_layers=cfg.gru_layers,
            batch_first=True,
            dropout=0.1 if cfg.gru_layers > 1 else 0.0,
        )
        self.long_encoder = nn.GRU(
            input_size=1,
            hidden_size=bh,
            num_layers=cfg.gru_layers,
            batch_first=True,
            dropout=0.1 if cfg.gru_layers > 1 else 0.0,
        )

        self.fusion = nn.Sequential(
            nn.Linear(bh * 3, bh * 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(bh * 2, bh),
            nn.SiLU(),
        )

        self.behavior_proj = nn.Sequential(
            nn.Linear(bh, bh),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(bh, bh),
        )

    def forward(
        self,
        short_seq: Tensor,   # (B, 120, 144)
        mid_seq: Tensor,     # (B, 240, 144)
        long_seq: Tensor,   # (B, 480, 144)
    ) -> Tensor:
        """Return B0: [B, base_behavior_dim] global market embedding."""
        _, h_short = self.short_encoder(short_seq)
        _, h_mid   = self.mid_encoder(mid_seq)
        _, h_long  = self.long_encoder(long_seq)

        # Take last layer's hidden for each GRU
        h_short = h_short[-1]  # (B, bh)
        h_mid   = h_mid[-1]
        h_long  = h_long[-1]

        combined = torch.cat([h_short, h_mid, h_long], dim=-1)
        fused = self.fusion(combined)
        B0 = self.behavior_proj(fused)

        return B0


# ══════════════════════════════════════════════════════════════════════════════
# Agent Behaviour Module (Level 2)
# ══════════════════════════════════════════════════════════════════════════════


class AgentBehaviorModule(nn.Module):
    """Generates per-path behavior embeddings from B0 + stochastic latents.

    For each of K paths:
        z_i  ~ N(0, I)         // independent latent
        B_i = f_agent(B0, z_i)  // modulated behavior

    The residual design (transform + agent_mod) ensures stability while
    the LayerNorm prevents collapse to zero-mean latents.
    """

    def __init__(self, config: BehaviorGenConfig):
        super().__init__()
        bh = config.base_behavior_dim   # 896
        ah = config.agent_behavior_dim  # 448
        self.agent_behavior_dim = ah    # store for forward()

        # Latent modulation network
        self.latent_net = nn.Sequential(
            nn.Linear(bh + ah, bh),
            nn.SiLU(),
            nn.Linear(bh, bh),
            nn.SiLU(),
            nn.Linear(bh, bh),
        )

        # Base transformation (residual branch)
        self.transform_net = nn.Sequential(
            nn.Linear(bh, bh),
            nn.SiLU(),
            nn.Linear(bh, bh),
        )

        self.norm = nn.LayerNorm(bh)

    def forward(
        self,
        B0: Tensor,
        num_paths: int,
        z: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Return (B_agent, z) where B_agent is (B, K, behavior_dim)."""
        B = B0.shape[0]

        if z is None:
            z = torch.randn(
                B, num_paths, self.agent_behavior_dim,
                device=B0.device, dtype=B0.dtype,
            )

        B0_expanded = B0.unsqueeze(1).expand(-1, num_paths, -1)  # (B, K, bh)

        combined = torch.cat([B0_expanded, z], dim=-1)           # (B, K, bh+ah)
        agent_mod = self.latent_net(combined)
        transformed = self.transform_net(B0_expanded)

        B_agent = self.norm(transformed + agent_mod)

        return B_agent, z


# ══════════════════════════════════════════════════════════════════════════════
# FiLM injector (shared utility)
# ══════════════════════════════════════════════════════════════════════════════


class FiLMInjection(nn.Module):
    """Feature-wise Linear Modulation:  FiLM(x; γ,β) = γ·x + β

    Learns (γ, β) from a conditioning vector. Used to inject behavior and
    timestep information into UNet blocks.
    """

    def __init__(self, condition_dim: int, feature_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(condition_dim, feature_dim * 2),
            nn.SiLU(),
        )

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        """x: (B, C, T),  condition: (B, condition_dim)"""
        params = self.net(condition)             # (B, 2*C)
        gamma, beta = params.chunk(2, dim=-1)   # (B, C) each
        gamma = gamma.unsqueeze(-1)             # (B, C, 1)
        beta  = beta.unsqueeze(-1)
        return x * (1.0 + gamma) + beta