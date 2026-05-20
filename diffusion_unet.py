"""True 1D temporal UNet with skip connections and local attention.

This is a complete rewrite — the old MMFPS_V2 "UNet" was just a linear
Conv1d stack. This implementation has:

  - Proper encoder-decoder hierarchy with skip concatenation
  - 2 down-blocks (stride-2 pooling) → 2 up-blocks (interpolation)
  - Local self-attention at the bottleneck (window=5)
  - FiLM conditioning at every depth (behavior + timestep)
  - Residual convolutions in every block
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import BehaviorGenConfig
from .behavioral_encoder import FiLMInjection, SinusoidalTimestepEmbedding, _make_timestep_mlp


# ── Local attention at bottleneck ──────────────────────────────────────────

class LocalTemporalAttention(nn.Module):
    """Sliding-window self-attention for 1D temporal sequences.

    Each position attends only to positions within `window` steps.
    Memory: O(B × T × window) instead of O(B × T²).
    """

    def __init__(self, channels: int, num_heads: int = 4, window: int = 5):
        super().__init__()
        assert channels % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.channels = channels
        self.window = window

        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.out_proj = nn.Conv1d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        B, C, T = x.shape
        H = self.num_heads
        D = self.head_dim

        qkv = self.qkv(x)                    # (B, 3C, T)
        q, k, v = qkv.chunk(3, dim=1)       # (B, C, T) each

        # Reshape to heads:  (B, H, D, T)
        q = q.view(B, H, D, T)
        k = k.view(B, H, D, T)
        v = v.view(B, H, D, T)

        # Compute attention with local window
        scale = 1.0 / math.sqrt(D)
        out = torch.zeros_like(q)

        w = self.window
        for start in range(0, T, w):
            end = min(start + w, T)
            q_w = q[:, :, :, start:end]                          # (B, H, D, w)

            # Attend over wider window to capture temporal context
            ctx_start = max(0, start - w)
            ctx_end = min(T, end + w)
            k_w = k[:, :, :, ctx_start:ctx_end]                  # (B, H, D, ctx_w)
            v_w = v[:, :, :, ctx_start:ctx_end]

            attn = torch.matmul(q_w.transpose(-1, -2), k_w) * scale  # (B, H, w, ctx_w)
            attn = F.softmax(attn, dim=-1)
            o_w = torch.matmul(attn, v_w.transpose(-1, -2))           # (B, H, w, D)

            out[:, :, :, start:end] = o_w.transpose(-1, -2)

        out = out.reshape(B, C, T)
        return self.out_proj(out)


# ── UNet building blocks ────────────────────────────────────────────────────

class ResConvBlock(nn.Module):
    """Conv1d → GroupNorm → SiLU → Conv1d → GroupNorm with residual.

    Uses pre-activation residual pattern for training stability.
    """

    def __init__(self, in_ch: int, out_ch: int, num_groups: int = 32):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups, in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        residual = self.skip(x)
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + residual


class DownBlock(nn.Module):
    """ResConv → FiLM-inj → stride-2 pool (halves temporal dim)."""

    def __init__(self, in_ch: int, out_ch: int, condition_dim: int):
        super().__init__()
        self.block = ResConvBlock(in_ch, out_ch)
        self.film = FiLMInjection(condition_dim, out_ch)
        self.pool = nn.Conv1d(out_ch, out_ch, 4, stride=2, padding=1)

    def forward(self, x: Tensor, condition: Tensor) -> tuple[Tensor, Tensor]:
        h = self.block(x)
        h = self.film(h, condition)
        h_pooled = self.pool(h)
        return h_pooled, h  # (pooled, full-res for skip)


class UpBlock(nn.Module):
    """Upsample → concat(skip) → ResConv → FiLM."""

    def __init__(self, in_ch: int, out_ch: int, condition_dim: int):
        super().__init__()
        self.conv_in = nn.Conv1d(in_ch * 2, out_ch, 1)
        self.block = ResConvBlock(out_ch, out_ch)
        self.film = FiLMInjection(condition_dim, out_ch)

    def forward(self, x: Tensor, skip: Tensor, condition: Tensor) -> Tensor:
        x_up = F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False)
        h = torch.cat([x_up, skip], dim=1)   # skip concat in channel dim
        h = self.conv_in(h)
        h = self.block(h)
        h = self.film(h, condition)
        return h


# ── Full UNet ────────────────────────────────────────────────────────────────

class DiffusionUNet(nn.Module):
    """1D temporal UNet with skip connections and local attention.

    Architecture:
        conv_in(1→base_ch)
        ├── down1(base_ch→base_ch)             ResConv×2 + FiLM + pool
        ├── down2(base_ch→ch×2)                ResConv×2 + FiLM + pool
        ├── mid(ch×2→ch×2)                     ResConv + LocalAttn + FiLM
        ├── up2(ch×2 → base_ch)               upsample + concat(skip2) + ResConv + FiLM
        ├── up1(base_ch → base_ch)            upsample + concat(skip1) + ResConv + FiLM
        └── conv_out(base_ch→1)

    Conditioning: behavior + timestep embeddings are projected to match
    each block's channel count and injected via FiLM at every depth.
    """

    def __init__(self, config: BehaviorGenConfig):
        super().__init__()
        cfg = config
        ch = cfg.base_channels   # 896
        cond_dim = ch             # condition dim = base channels (timestep + behavior)

        # Timestep embedding: sinusoidal → MLP → (B, ch)
        self.time_embed = SinusoidalTimestepEmbedding(cfg.base_behavior_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(cfg.base_behavior_dim, ch * 2),
            nn.SiLU(),
            nn.Linear(ch * 2, ch),
        )

        # Behavior projector: behavior_dim → ch (to add to timestep)
        self.behavior_proj = nn.Sequential(
            nn.Linear(cfg.base_behavior_dim, ch * 2),
            nn.SiLU(),
            nn.Linear(ch * 2, ch),
        )

        # ── Encoder ────────────────────────────────────────────────
        self.conv_in = nn.Conv1d(cfg.path_feature_dim, ch, 3, padding=1)

        self.down1 = DownBlock(ch, ch, cond_dim)          # ch → ch,  half temporal
        self.down2 = DownBlock(ch, ch * 2, cond_dim)       # ch → 2ch, half temporal

        self.mid_block = ResConvBlock(ch * 2, ch * 2)
        self.mid_attn = LocalTemporalAttention(
            ch * 2, num_heads=4, window=cfg.attention_window
        )
        self.mid_film = FiLMInjection(cond_dim, ch * 2)  # cond_dim → film(ch*2)

        # ── Decoder ────────────────────────────────────────────────
        self.up2 = UpBlock(ch * 2, ch, cond_dim)          # 2ch → ch (skip from down2)
        self.up1 = UpBlock(ch, ch, cond_dim)              # ch  → ch (skip from down1)

        self.conv_out = nn.Conv1d(ch, cfg.path_feature_dim, 1)

        # Initialize final conv with small weights for stability
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(
        self,
        noisy_returns: Tensor,      # (B_total, 1, T) — flattened paths
        timestep: Tensor,           # (B_total,) or None if inference
        behavior: Tensor,           # (B_total, base_behavior_dim) — per-sample behavior
        debug: bool = False,
    ) -> Tensor:
        """Denoise predict:  ε_θ(noisy_returns, t, behavior).

        Returns predicted noise of same shape as noisy_returns.
        Set debug=True to print shapes during smoke tests.
        """
        B_total, _, T = noisy_returns.shape
        ch = self.conv_in.out_channels

        # ── Build conditioning vector ──────────────────────────────
        # Timestep: sinusoidal → MLP → (B_total, ch)
        if timestep is not None:
            t_emb = self.time_embed(timestep)          # (B_total, 896)
            cond_t = self.time_proj(t_emb)             # (B_total, ch)
        else:
            cond_t = torch.zeros(B_total, ch, device=noisy_returns.device)

        # Behavior: behavior_dim → MLP → (B_total, ch)
        cond_b = self.behavior_proj(behavior)          # (B_total, ch)

        # Combine: add timestep + behavior conditioning
        combined_cond = cond_t + cond_b                # (B_total, ch)

        if debug:
            print(f"  UNet: noisy={noisy_returns.shape}, t={timestep.shape if timestep is not None else None}, beh={behavior.shape}")
            print(f"  UNet: cond_t={cond_t.shape}, cond_b={cond_b.shape}, combined={combined_cond.shape}")

        # ── Encoder ────────────────────────────────────────────────
        h = self.conv_in(noisy_returns)                 # (B_total, ch, T)
        if debug: print(f"  UNet conv_in: {h.shape}")

        h_down1, skip1 = self.down1(h, combined_cond)  # pool: T→T/2, ch→ch
        if debug: print(f"  UNet down1: h={h_down1.shape}, skip={skip1.shape}")

        h_down2, skip2 = self.down2(h_down1, combined_cond)  # pool: T/2→T/4, ch→2ch
        if debug: print(f"  UNet down2: h={h_down2.shape}, skip={skip2.shape}")

        # ── Bottleneck ─────────────────────────────────────────────
        h_mid = self.mid_block(h_down2)                 # (B_total, 2ch, T/4)
        h_attn = self.mid_attn(h_mid)                   # same shape
        h_mid = h_mid + h_attn                          # residual attention
        h_mid = self.mid_film(h_mid, combined_cond)     # FiLM injection
        if debug: print(f"  UNet mid: {h_mid.shape}")

        # ── Decoder ────────────────────────────────────────────────
        h_up2 = self.up2(h_mid, skip2, combined_cond)  # upsample+skip: T/4→T/2, 2ch→ch
        if debug: print(f"  UNet up2: {h_up2.shape}")

        h_up1 = self.up1(h_up2, skip1, combined_cond)  # upsample+skip: T/2→T, ch→ch
        if debug: print(f"  UNet up1: {h_up1.shape}")

        out = self.conv_out(h_up1)                     # (B_total, 1, T)
        if debug: print(f"  UNet conv_out: {out.shape}")

        return out