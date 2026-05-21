"""Visualization callbacks for MMFPS_GEN_V2 training monitoring.

Saves per-step plots:
  - Multi-path overlay (all generated paths vs real target)
  - Closest manifold future (best-matching path)
  - Path correlation heatmap
  - Denoising progression (snapshots of diffusion reverse process)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from generator import BehaviorDiffusionGenerator


def _setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


@torch.no_grad()
def viz_generate(
    model: BehaviorDiffusionGenerator,
    short_seq: Tensor,
    mid_seq: Tensor,
    long_seq: Tensor,
    target: np.ndarray,
    output_dir: Path,
    sample_idx: int = 0,
    num_paths: int = 128,
    device: str = "cuda",
) -> None:
    """Generate and save comprehensive visualization for one sample."""
    plt = _setup_matplotlib()
    device_t = torch.device(device)

    short = short_seq.unsqueeze(0).to(device_t)
    mid = mid_seq.unsqueeze(0).to(device_t)
    long = long_seq.unsqueeze(0).to(device_t)

    paths, behaviors, base_beh = model.generate(
        short, mid, long, num_paths=num_paths
    )
    paths_np = paths[0].cpu().numpy()   # (num_paths, T)
    beh_np = behaviors[0].cpu().numpy()  # (num_paths, D)

    # Find closest path
    sq_err = ((paths_np - target) ** 2).mean(axis=-1)
    best_idx = sq_err.argmin()
    closest_path = paths_np[best_idx]

    output_dir.mkdir(parents=True, exist_ok=True)
    T = paths_np.shape[1]

    # ── Plot 1: Multi-path overlay ───────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    for p in paths_np:
        ax.plot(p, alpha=0.12, color="steelblue", linewidth=0.4)
    ax.plot(target, "r-", linewidth=2, label="Real Future")
    ax.plot(closest_path, "g--", linewidth=1.5, label=f"Closest (idx={best_idx})")
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.3)
    ax.set_xlabel("Step")
    ax.set_ylabel("Return")
    ax.set_title(f"Generated Returns — {num_paths} paths")
    ax.legend()
    fig.savefig(output_dir / f"overlay_s{sample_idx:04d}.png", dpi=100)
    plt.close(fig)

    # ── Plot 2: Closest path detail ──────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(target, "r-o", linewidth=1.5, markersize=4, label="Real")
    ax.plot(closest_path, "g-s", linewidth=1.5, markersize=4, label=f"Closest (MSE={sq_err[best_idx]:.6f})")
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.3)
    ax.set_xlabel("Step")
    ax.set_ylabel("Return")
    ax.set_title(f"Closest Manifold Future (idx={best_idx})")
    ax.legend()
    fig.savefig(output_dir / f"closest_s{sample_idx:04d}.png", dpi=100)
    plt.close(fig)

    # ── Plot 3: Path correlation heatmap ─────────────────────────
    corr = np.corrcoef(paths_np[:50])  # Top 50 paths
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xlabel("Path Index")
    ax.set_ylabel("Path Index")
    ax.set_title("Path Correlation Matrix (Top 50)")
    plt.colorbar(im, ax=ax, label="Correlation")
    fig.savefig(output_dir / f"corr_s{sample_idx:04d}.png", dpi=100)
    plt.close(fig)

    # ── Plot 4: Endpoint distribution ─────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    endpoints = paths_np[:, -1]
    ax.hist(endpoints, bins=40, color="steelblue", alpha=0.7, edgecolor="white")
    ax.axvline(x=target[-1], color="red", linewidth=2, label="Real Final")
    ax.axvline(x=0, color="gray", linestyle=":", alpha=0.3)
    ax.set_xlabel("Final Return")
    ax.set_ylabel("Count")
    ax.set_title("Endpoint Distribution")
    ax.legend()
    fig.savefig(output_dir / f"endpoints_s{sample_idx:04d}.png", dpi=100)
    plt.close(fig)


def viz_denoising(
    model: BehaviorDiffusionGenerator,
    short_seq: Tensor,
    mid_seq: Tensor,
    long_seq: Tensor,
    device: str = "cuda",
    num_snapshots: int = 6,
    output_path: Optional[Path] = None,
) -> None:
    """Visualize the DDIM denoising process step by step."""
    plt = _setup_matplotlib()
    device_t = torch.device(device)

    B0 = model.base_encoder(short_seq.unsqueeze(0).to(device_t),
                            mid_seq.unsqueeze(0).to(device_t),
                            long_seq.unsqueeze(0).to(device_t))
    B_agent, _ = model.agent_module(B0, 1)

    scheduler = model.scheduler
    T = model.config.path_horizon
    timesteps = scheduler._get_inference_timesteps(device_t)
    step_indices = np.linspace(0, len(timesteps) - 1, num_snapshots, dtype=int)

    x_t = torch.randn(1, 1, T, device=device_t)
    snapshots = []

    for i, t in enumerate(timesteps):
        t_batch = t.unsqueeze(0).expand(1)
        eps_pred = model.unet(x_t, t_batch, B_agent[:, 0, :])
        eps_pred = eps_pred.reshape(1, 1, T)
        x_t = scheduler._ddim_step(x_t, eps_pred, t.item(), i < len(timesteps) - 1)
        if i in step_indices:
            snapshots.append((i, x_t[0, 0].cpu().numpy().copy()))

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for ax, (si, path) in zip(axes, snapshots):
        ax.plot(path, "b-", linewidth=1.5)
        ax.axhline(y=0, color="gray", linestyle=":", alpha=0.3)
        ax.set_title(f"t = {timesteps[si].item()}")
    fig.suptitle("DDIM Denoising Progression")
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=100)
        plt.close(fig)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Visualize MMFPS_GEN_V2 outputs")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="viz")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--num-paths", type=int, default=128)
    parser.add_argument("--data-index", type=int, default=0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = BehaviorDiffusionGenerator(ckpt.get("config")).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    from dataset import PathDataset
    ds = PathDataset(model.config, split="val")

    for i in range(args.num_samples):
        sample = ds[args.data_index + i]
        target = sample.target.numpy()
        viz_generate(
            model,
            sample.short_seq, sample.mid_seq, sample.long_seq,
            target, output_dir,
            sample_idx=i, num_paths=args.num_paths, device=device,
        )

    print(f"Visualizations saved to {output_dir}")

    # Denoising viz for one sample
    sample = ds[args.data_index]
    viz_denoising(
        model,
        sample.short_seq, sample.mid_seq, sample.long_seq,
        device=device,
        output_path=output_dir / "denoising_progression.png",
    )
    print(f"Denoising progression saved to {output_dir / 'denoising_progression.png'}")


if __name__ == "__main__":
    main()