"""Quick forward-pass debug script."""
import torch
from config import BehaviorGenConfig
from generator import BehaviorDiffusionGenerator

cfg = BehaviorGenConfig()
cfg.base_channels = 512
cfg.batch_size = 2

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Creating model...")
model = BehaviorDiffusionGenerator(cfg).to(device)
print(f"Model created. Params: {model.count_parameters():,}")

short = torch.randn(2, cfg.short_horizon, cfg.feature_dim).to(device)
mid = torch.randn(2, cfg.mid_horizon, cfg.feature_dim).to(device)
long = torch.randn(2, cfg.long_horizon, cfg.feature_dim).to(device)

print("Running forward...")

with torch.no_grad():
    out = model.generate(
        short_seq=short,
        mid_seq=mid,
        long_seq=long,
        num_paths=4,
    )

print("SUCCESS")
print(f"Output paths: {out[0].shape}")
