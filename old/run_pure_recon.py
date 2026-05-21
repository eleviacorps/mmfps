"""Pure reconstruction launcher — no structural losses, max throughput."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from MMFPS_GEN_V2.config import BehaviorGenConfig
from MMFPS_GEN_V2.trainer import train

DATA = "D:/Programming/AiProjects/Nexus-MMFPS/MMFPS_GEN_V2/main_bars_data/diffusion_fused_6m.npy"
OUT  = "D:/Programming/AiProjects/Nexus-MMFPS/MMFPS_GEN_V2/checkpoints/pure_recon"

cfg = BehaviorGenConfig(
    batch_size=32,
    learning_rate=5e-5,
    weight_volatility=0.0,
    weight_trend=0.0,
    weight_turning=0.0,
    weight_dtw=0.0,
    weight_diversity=0.0,
    weight_latent_sensitivity=0.0,
    weight_manifold_spread=0.0,
    training_paths_per_sample=16,
    log_every=10,
    checkpoint_every=1000,
    visualize_every=2500,
)
cfg._total_steps = 5000

if __name__ == "__main__":
    train(data_path=DATA, output_dir=OUT, config=cfg)
