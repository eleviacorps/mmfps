"""Stage 2: resume from pure reconstruction checkpoint with mild structural losses."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from MMFPS_GEN_V2.config import BehaviorGenConfig
from MMFPS_GEN_V2.trainer import train

DATA = "D:/Programming/AiProjects/Nexus-MMFPS/MMFPS_GEN_V2/main_bars_data/diffusion_fused_6m.npy"
OUT  = "D:/Programming/AiProjects/Nexus-MMFPS/MMFPS_GEN_V2/checkpoints/pure_recon"

# Resume from step 4000, add gentle structural losses
RESUME = "D:/Programming/AiProjects/Nexus-MMFPS/MMFPS_GEN_V2/checkpoints/pure_recon/step_4000.pt"

cfg = BehaviorGenConfig(
    batch_size=32,
    learning_rate=5e-5,
    weight_volatility=0.0,
    weight_trend=0.02,         # mild trend-following pressure
    weight_turning=0.0,
    weight_dtw=0.0,
    weight_diversity=0.002,    # mild anti-collapse
    weight_latent_sensitivity=0.0,
    weight_manifold_spread=0.0,
    training_paths_per_sample=16,
    log_every=10,
    checkpoint_every=1000,
    visualize_every=2500,
)
cfg._total_steps = 8000  # 4000 more from current step 4000

if __name__ == "__main__":
    train(data_path=DATA, output_dir=OUT, config=cfg, resume_from=RESUME)
