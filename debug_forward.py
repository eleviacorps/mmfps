import torch
from Nexus_Packaged.MMFPS_GEN_V2.config import BehaviorGenConfig
from Nexus_Packaged.MMFPS_GEN_V2.generator import BehaviorDiffusionGenerator

cfg = BehaviorGenConfig()

cfg.base_channels = 512
cfg.batch_size = 2

device = "cuda"

print("Creating model...")
model = BehaviorDiffusionGenerator(cfg).to(device)

print("Model created.")

short = torch.randn(2, cfg.short_horizon, 1).to(device)
mid = torch.randn(2, cfg.mid_horizon, 1).to(device)
long = torch.randn(2, cfg.long_horizon, 1).to(device)

print("Running forward...")

with torch.no_grad():
    out = model.generate(
        short_seq=short,
        mid_seq=mid,
        long_seq=long,
        num_paths=4,
    )

print("SUCCESS")
print(out.shape)