"""Continue pure reconstruction from step 4000 → 5000 (seamless LR continuation)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from MMFPS_GEN_V2.config import BehaviorGenConfig
from MMFPS_GEN_V2.trainer import train

DATA = "D:/Programming/AiProjects/Nexus-MMFPS/MMFPS_GEN_V2/main_bars_data/diffusion_fused_6m.npy"
OUT  = "D:/Programming/AiProjects/Nexus-MMFPS/MMFPS_GEN_V2/checkpoints/pure_recon"
RESUME = "D:/Programming/AiProjects/Nexus-MMFPS/MMFPS_GEN_V2/checkpoints/pure_recon/step_4000.pt"

cfg = BehaviorGenConfig(
    batch_size=32,
    learning_rate=5e-5,
    training_paths_per_sample=16,
    log_every=10,
    checkpoint_every=500,
    visualize_every=5000,
)
cfg._total_steps = 5000  # same as original run → seamless LR continuation

if __name__ == "__main__":
    train(data_path=DATA, output_dir=OUT, config=cfg, resume_from=RESUME)
