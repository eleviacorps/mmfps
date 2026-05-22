"""Unified configuration for MMFPS_GEN_V2 Behavioral Diffusion Generator."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BehaviorGenConfig:
    """All hyperparameters for the MMFPS_GEN_V2 generator.

    Organized into logical groups so each can be tuned independently.
    """

    # ── Feature / data dimensions ────────────────────────────────────────────
    feature_dim: int = 42         # Derived features (pre-computed)
    path_feature_dim: int = 1       # Single scalar per timestep (returns space)

    # ── Temporal horizons (in bars) ──────────────────────────────────────────
    short_horizon: int = 32         # Local momentum window (last 32 of 128)
    mid_horizon: int = 64           # Medium regime window (last 64 of 128)
    long_horizon: int = 128         # Macro structure window (full window)
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
    sampling_noise_scale: float = 1.0      # Initial DDIM noise std (must match training distribution)

    # ── Loss weights ───────────────────────────────────────────────────────────
    weight_mse: float = 1.0
    weight_volatility: float = 0.5
    weight_trend: float = 0.1
    weight_turning: float = 0.05
    weight_dtw: float = 0.1
    weight_smoothness: float = 0.02
    weight_autocorr: float = 0.02
    weight_tail: float = 0.01
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

    # ── Target normalization ───────────────────────────────────────────────────
    target_scale: float = 1.0          # Multiplier to bring targets to unit variance
                                       # (auto-computed from training data in build_splits)
    min_target_std: float = 1e-8        # Reject stale/flat future windows

    # ── Data paths ─────────────────────────────────────────────────────────────
    data_dir: str = "data"
    features_file: str = "normalized_feature_tensor.npy"
    targets_file: str = "target_tensor.npy"
    target_valid_file: str = "target_valid.npy"
    session_id_file: str = "session_id.npy"
    timestamps_file: str = "feature_timestamps.npy"

    # ── Data splits (timestamps in seconds) ────────────────────────────────────
    val_split_ts: int = 1672531200    # 2023-01-01 00:00:00 UTC
    test_split_ts: int = 1735689600   # 2025-01-01 00:00:00 UTC


# ── Global default instance ───────────────────────────────────────────────────
DEFAULT_CONFIG = BehaviorGenConfig()
