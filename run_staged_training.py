"""Staged training launcher for MMFPS_GEN_V2 behavioral emergence validation.

Progression: Stage A (500 steps) → Stage B (2000 steps) → Stage C (5000 steps)

After each stage: save checkpoint, run evaluation, generate visualizations.
This validates manifold emergence, not just scalar loss reduction.

Usage:
    python -m Nexus_Packaged.MMFPS_GEN_V2.run_staged_training [--data-path PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch

# Ensure package root is on path
_pkg = Path(__file__).resolve().parents[1]  # Nexus_Packaged directory
if str(_pkg) not in sys.path:
    sys.path.insert(0, str(_pkg))

from MMFPS_GEN_V2.config import BehaviorGenConfig
from MMFPS_GEN_V2.generator import BehaviorDiffusionGenerator
from MMFPS_GEN_V2.trainer import (
    train,
    _save_checkpoint,
    _load_checkpoint,
    _viz_callback,
    EMAWrapper,
)
from MMFPS_GEN_V2.dataset import build_splits
from MMFPS_GEN_V2.evaluate import evaluate as run_evaluation


# ── Stage definitions ───────────────────────────────────────────────────────

STAGES = {
    "A": {
        "total_steps": 1000,
        "description": "Pure denoising stabilization — no manifold/diversity/trend pressure, EMA disabled",
        "loss_overrides": {
            "weight_trend": 0.0,
            "weight_turning": 0.0,
            "weight_dtw": 0.0,
            "weight_diversity": 0.0,
            "weight_manifold_spread": 0.0,
            "weight_latent_sensitivity": 0.0,
        },
        "config_overrides": {
            "training_paths_per_sample": 4,
        },
        "ema_enabled": False,
    },
    "B": {
        "total_steps": 2000,
        "description": "Structural emergence — DDIM refinement, moderate structural losses",
        "loss_overrides": {
            "weight_trend": 0.05,
            "weight_turning": 0.02,
            "weight_dtw": 0.0,
            "weight_diversity": 0.001,
            "weight_manifold_spread": 0.005,
        },
        "config_overrides": {
            "training_paths_per_sample": 8,
        },
    },
    "C": {
        "total_steps": 5000,
        "description": "Behavior validation — diversity stability, latent control",
        "loss_overrides": {
            "weight_trend": 0.1,
            "weight_turning": 0.05,
            "weight_dtw": 0.0,
            "weight_diversity": 0.005,
            "weight_manifold_spread": 0.01,
        },
        "config_overrides": {
            "training_paths_per_sample": 16,
        },
    },
    "D": {
        "total_steps": 10000,
        "description": "Extended training — full manifold quality assessment",
        "loss_overrides": {
            "weight_trend": 0.1,
            "weight_turning": 0.05,
            "weight_dtw": 0.0,
            "weight_diversity": 0.01,
            "weight_manifold_spread": 0.02,
        },
        "config_overrides": {
            "training_paths_per_sample": 16,
        },
    },
}


def _resolve_device() -> torch.device:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Try MPS (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    print("\n[WARNING] No GPU detected. Training 151M params on CPU will be SLOW.")
    print("Consider installing a CUDA-enabled PyTorch.")
    return torch.device("cpu")


def _run_stage(
    stage_name: str,
    data_path: str,
    output_dir: Path,
    config: BehaviorGenConfig,
    device: torch.device,
    resume_from: Optional[Path] = None,
) -> Path:
    """Run one training stage and return path to final checkpoint."""
    stage = STAGES[stage_name]
    print(f"\n{'=' * 60}")
    print(f"  STAGE {stage_name}: {stage['total_steps']} steps")
    print(f"  {stage['description']}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 60}\n")

    # Apply stage-specific overrides
    for k, v in stage.get("loss_overrides", {}).items():
        setattr(config, k, v)
    for k, v in stage.get("config_overrides", {}).items():
        setattr(config, k, v)

    print(f"  Stage loss weights:")
    for key in ["weight_mse", "weight_volatility", "weight_trend", "weight_turning",
                 "weight_diversity", "weight_manifold_spread", "weight_latent_sensitivity"]:
        print(f"    {key} = {getattr(config, key, 'N/A')}")
    print(f"    base_channels = {config.base_channels}")
    print(f"    training_paths_per_sample = {config.training_paths_per_sample}")

    # Configure vis/log frequencies based on stage length
    total = stage["total_steps"]
    config._total_steps = total
    config._max_epochs = 1000  # effectively unlimited, stopped by _total_steps
    config.checkpoint_every = max(total // 5, 250)
    config.visualize_every = max(total // 5, 250)
    config.log_every = 10

    stage_dir = output_dir / f"stage_{stage_name}"
    stage_dir.mkdir(parents=True, exist_ok=True)

    # Run training
    ema_enabled = stage.get("ema_enabled", True)
    model = train(
        data_path=data_path,
        output_dir=str(stage_dir),
        config=config,
        resume_from=str(resume_from) if resume_from else None,
        device=str(device),
        ema_enabled=ema_enabled,
    )

    # Find latest checkpoint
    checkpoints = sorted(stage_dir.glob("step_*.pt"))
    if not checkpoints:
        checkpoints = sorted(stage_dir.glob("*.pt"))
    latest = checkpoints[-1] if checkpoints else None

    if latest is None:
        raise RuntimeError(f"No checkpoint saved after Stage {stage_name}!")

    print(f"\nStage {stage_name} complete. Checkpoint: {latest}")

    # ── Run evaluation ───────────────────────────────────────────
    print(f"\nRunning evaluation on Stage {stage_name}...")
    eval_results = run_evaluation(
        checkpoint_path=str(latest),
        data_path=data_path,
        config=config,
        num_paths=128,
        max_samples=500,
        device=str(device),
    )

    eval_path = stage_dir / f"eval_stage_{stage_name}.json"
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2)

    print(f"\n  Generator success rate: {eval_results['generator_success_rate']:.2%}")
    print(f"  Cone coverage:          {eval_results['cone_coverage_rate']:.2%}")
    print(f"  Mean closest distance:  {eval_results['mean_closest_distance']:.6f}")
    print(f"  Trending success:       {eval_results['trending_success_rate']:.2%}")
    print(f"  Ranging success:        {eval_results['ranging_success_rate']:.2%}")
    print(f"  Evaluation saved to:    {eval_path}")

    # ── Generate final visualizations ────────────────────────────
    print(f"\nGenerating visualizations...")
    _, val_ds, _ = build_splits(data_path, config)
    _viz_callback(
        model, val_ds, stage_dir / "viz", stage["total_steps"],
        device, num_samples=4,
    )
    print(f"  Visualizations saved to: {stage_dir / 'viz'}\n")

    return latest


def main():
    parser = argparse.ArgumentParser(
        description="MMFPS_GEN_V2 — Staged Training Validation"
    )
    parser.add_argument(
        "--data-path", type=str,
        default="D:/Programming/AiProjects/Nexus-MMFPS/MMFPS_GEN_V2/main_bars_data/diffusion_fused_6m.npy",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="D:/Programming/AiProjects/Nexus-MMFPS/MMFPS_GEN_V2/checkpoints",
    )
    parser.add_argument(
        "--stage", type=str, default="A",
        choices=["A", "B", "C", "D", "all"],
        help="Which stage to run (A=500, B=2000, C=5000, D=10000, all=sequential A→D)",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Resume from an existing checkpoint")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print config and exit without training")
    args = parser.parse_args()

    device = _resolve_device()
    cfg = BehaviorGenConfig(
        batch_size=args.batch_size,
        learning_rate=args.lr,
        visualize_every=250,
        checkpoint_every=500,
        log_every=10,
        max_samples=0,  # use all available data
    )

    print(f"Device: {device}")
    print(f"Data:   {args.data_path}")
    print(f"Stage:  {args.stage}")
    print(f"Config: bs={cfg.batch_size}, lr={cfg.learning_rate}, "
          f"ch={cfg.base_channels}, paths/train={cfg.training_paths_per_sample}")

    if args.dry_run:
        print("\n[Dry run — exiting]")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config for reproducibility
    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps({
        k: v for k, v in cfg.__dict__.items()
        if not k.startswith("_") and not callable(v)
    }, indent=2, default=str))

    resume = Path(args.resume_from) if args.resume_from else None

    if args.stage == "all":
        current = resume
        for s in ["A", "B", "C", "D"]:
            current = _run_stage(s, args.data_path, output_dir, cfg, device, current)
    else:
        _run_stage(args.stage, args.data_path, output_dir, cfg, device, resume)

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == "__main__":
    main()