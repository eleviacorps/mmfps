"""MMFPS_GEN_V2 — Behavioral Diffusion Future Manifold Generator v2.0.0.

This is the COMPLETE REBUILD of the MMFPS generator with:
  - True iterative DDIM diffusion sampling
  - Proper UNet with skip connections + local attention
  - Correct diversity losses (anti-collapse)
  - DTW for 100% of samples
  - Temporal train/val/test split
  - Mixed-precision training with EMA
  - Real-time manifold diagnostics
  - Visualization callbacks

Architecture:
  Context → BaseBehaviorEncoder → B0
  B0 + z → AgentBehaviorModule → B_agent
  B_agent + noise → DiffusionUNet → ε_pred
  ε_pred → DDIMScheduler (50 steps) → 128 return paths

Usage:
  from Nexus_Packaged.MMFPS_GEN_V2 import BehaviorDiffusionGenerator, BehaviorGenConfig
  model = BehaviorDiffusionGenerator()
  paths, behaviors, _ = model.generate(short, mid, long, num_paths=128)
"""

from .config import BehaviorGenConfig
from .behavioral_encoder import (
    BaseBehaviorEncoder,
    AgentBehaviorModule,
    SinusoidalTimestepEmbedding,
    FiLMInjection,
)
from .diffusion_unet import DiffusionUNet, LocalTemporalAttention
from .diffusion_sampler import DDIMScheduler
from .generator import BehaviorDiffusionGenerator
from .losses import compute_all_losses
from .dataset import PathDataset, build_splits, Sample
from .trainer import train, EMAWrapper

__version__ = "2.0.0"
__all__ = [
    "BehaviorGenConfig",
    "BaseBehaviorEncoder",
    "AgentBehaviorModule",
    "SinusoidalTimestepEmbedding",
    "FiLMInjection",
    "DiffusionUNet",
    "LocalTemporalAttention",
    "DDIMScheduler",
    "BehaviorDiffusionGenerator",
    "compute_all_losses",
    "PathDataset",
    "build_splits",
    "Sample",
    "train",
    "EMAWrapper",
]