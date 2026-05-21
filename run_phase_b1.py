"""Phase B1: tiny structural refinement after pure diffusion emergence.

Starts from the pure reconstruction checkpoint as weights only, with a fresh
optimizer and LR schedule. This avoids inheriting Stage A's zero-LR final
scheduler state.
"""

from pathlib import Path

from config import BehaviorGenConfig
from trainer import train

OUT = Path(__file__).parent / "checkpoints" / "phase_b1"
RESUME = Path(__file__).parent / "checkpoints" / "pure_recon" / "step_5000_final.pt"

cfg = BehaviorGenConfig(
    batch_size=32,
    learning_rate=1e-5,
    warmup_steps=200,
    weight_volatility=0.03,
    weight_trend=0.0,
    weight_turning=0.01,
    weight_dtw=0.0,
    weight_smoothness=0.0,
    weight_autocorr=0.0,
    weight_tail=0.0,
    weight_diversity=0.0,
    weight_latent_sensitivity=0.0,
    weight_manifold_spread=0.0,
    training_paths_per_sample=16,
    log_every=10,
    checkpoint_every=500,
    visualize_every=1000,
)
cfg._total_steps = 1500

if __name__ == "__main__":
    train(
        output_dir=str(OUT),
        config=cfg,
        resume_from=str(RESUME),
        resume_weights_only=True,
    )
