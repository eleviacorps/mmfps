"""Continue pure reconstruction from step 4000 (seamless LR continuation)."""
from pathlib import Path

from config import BehaviorGenConfig
from trainer import train

OUT = Path(__file__).parent / "checkpoints" / "pure_recon"
RESUME = str(OUT / "step_4000.pt")

cfg = BehaviorGenConfig(
    batch_size=32,
    learning_rate=5e-5,
    weight_volatility=0.0,
    weight_trend=0.0,
    weight_turning=0.0,
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
    visualize_every=5000,
)
cfg._total_steps = 5000

if __name__ == "__main__":
    train(output_dir=str(OUT), config=cfg, resume_from=RESUME)
