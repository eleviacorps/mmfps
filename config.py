"""Unified configuration for MMFPS_GEN_V2 Behavioral Diffusion Generator."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BehaviorGenConfig:
    """All hyperparameters for the MMFPS_GEN_V2 generator.

    Organized into logical groups so each can be tuned independently.
    """

    # ── Feature / data dimensions ────────────────────────────────────────────
    feature_dim: int = 8          # Feature dimension (derived features from price)
    path_feature_dim: int = 1       # Single scalar per timestep (returns space)

    # ── Temporal horizons (in bars) ──────────────────────────────────────────
    short_horizon: int = 120        # Local momentum window
    mid_horizon: int = 240          # Medium regime window
    long_horizon: int = 480         # Macro structure window
    path_horizon: int = 20          # Future to generate (target horizon)

    # ── UNet architecture ─────────────────────────────────────────────────────
    base_channels: int = 256        # Channel count after conv_in (~8M model)
    num_res_blocks: int = 2         # Conv blocks per down/up stage
    attention_window: int = 5       # Local attention window size at bottleneck

    # ── Behavioral embeddings ─────────────────────────────────────────────────
    base_behavior_dim: int = 256    # B0: global market behavior
    agent_behavior_dim: int = 128    # zi: per-path latent dimension
    gru_layers: int = 2             # GRU layers per horizon encoder

    # ── Diffusion ─────────────────────────────────────────────────────────────
    diffusion_timesteps: int = 100          # Training noise schedule length
    sampling_steps: int = 20                # DDIM reverse steps (can be < timesteps)
    noise_scale: float = 0.01               # Stddev of initial sampling noise
    sampling_eta: float = 0.0               # DDIM eta (0=DDIM, 1=DDPM)

    # ── Generation ────────────────────────────────────────────────────────────
    num_paths: int = 128            # Futures to generate per sample
    training_paths_per_sample: int = 16   # Paths per sample during training (memory)
    sampling_noise_scale: float = 0.25     # Scale factor on DDIM sampling noise

    # ── Loss weights ───────────────────────────────────────────────────────────
    weight_mse: float = 1.0
    weight_volatility: float = 0.5
    weight_trend: float = 0.1
    weight_turning: float = 0.05
    weight_dtw: float = 0.1
    weight_diversity: float = 0.01         # Pairwise path repulsion
    weight_latent_sensitivity: float = 0.05
    weight_manifold_spread: float = 0.02  # Anti-collapse via endpoint spread

    # ── Training ───────────────────────────────────────────────────────────────
    batch_size: int = 16
    accumulation_steps: int = 1      # Gradient accumulation (effective_bs = bs×acc)
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 2000
    max_grad_norm: float = 10.0
    ema_decay: float = 0.9999
    num_workers: int = 0             # Data loader workers (0=main thread, avoids Windows spawn overhead)

    # ── Checkpoints / logging ──────────────────────────────────────────────────
    checkpoint_every: int = 1000
    log_every: int = 10
    visualize_every: int = 500
    max_samples: int = 6_000_000      # Cap on loaded samples (0 = unlimited)

        # ── Bounded loss thresholds ────────────────────────────────────────
    # Margin-based formulation: penalty only if metric below threshold
    diversity_min_distance: float = 0.25           # Minimum mean pairwise path distance
    manifold_min_spread: float = 0.20              # Minimum endpoint std dev
    latent_sensitivity_min_distance: float = 0.15  # Minimum latent-variation translation

    # ── Data ─────────────────────────────────────────────────────────────────
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1           # train+val+test should sum to 1.0


# ── Global default instance ───────────────────────────────────────────────────
DEFAULT_CONFIG = BehaviorGenConfig()