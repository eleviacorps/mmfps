"""Emergence dashboard for pure diffusion validation.

This is a read-only checkpoint analysis tool. It does not train, tune, or add
losses; it helps answer whether pure diffusion is producing coherent stochastic
financial futures before structural regularization is introduced.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from config import BehaviorGenConfig
from dataset import PathDataset
from generator import BehaviorDiffusionGenerator


def _load_model(checkpoint: Path, device: torch.device) -> BehaviorDiffusionGenerator:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("config", BehaviorGenConfig())
    model = BehaviorDiffusionGenerator(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.scheduler.noise_scale_val = 1.0
    return model


def _moments(x: np.ndarray) -> dict[str, float]:
    flat = np.asarray(x, dtype=np.float64).reshape(-1)
    mean = float(flat.mean())
    std = float(flat.std() + 1e-12)
    centered = (flat - mean) / std
    hist, _ = np.histogram(flat, bins=80, density=True)
    p = hist / (hist.sum() + 1e-12)
    p = p[p > 0]
    return {
        "mean": mean,
        "std": std,
        "skew": float(np.mean(centered**3)),
        "kurtosis": float(np.mean(centered**4) - 3.0),
        "entropy": float(-(p * np.log(p)).sum()),
        "tail_p95_abs": float(np.percentile(np.abs(flat), 95)),
        "tail_p99_abs": float(np.percentile(np.abs(flat), 99)),
    }


def _lag_autocorr(x: np.ndarray, max_lag: int = 10) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    vals = []
    for lag in range(1, max_lag + 1):
        if arr.shape[-1] <= lag:
            vals.append(0.0)
            continue
        a = arr[:, :-lag] - arr[:, :-lag].mean(axis=-1, keepdims=True)
        b = arr[:, lag:] - arr[:, lag:].mean(axis=-1, keepdims=True)
        denom = np.sqrt((a * a).mean(axis=-1) * (b * b).mean(axis=-1)) + 1e-12
        vals.append(float(((a * b).mean(axis=-1) / denom).mean()))
    return np.array(vals)


def _turning_frequency(x: np.ndarray) -> float:
    arr = np.asarray(x)
    if arr.ndim == 1:
        arr = arr[None, :]
    signs = np.sign(np.diff(arr, axis=-1))
    turns = signs[:, 1:] * signs[:, :-1] < 0
    return float(turns.mean()) if turns.size else 0.0


def _max_drawdown(path: np.ndarray) -> np.ndarray:
    equity = np.cumsum(path, axis=-1)
    peak = np.maximum.accumulate(equity, axis=-1)
    return (peak - equity).max(axis=-1)


def _sample_with_snapshots(model, short, mid, long, num_paths: int, seed: int):
    torch.manual_seed(seed)
    scheduler = model.scheduler
    device = short.device
    T = model.config.path_horizon
    B0 = model.base_encoder(short, mid, long)
    B_agent, _ = model.agent_module(B0, num_paths)
    x_t = torch.randn(1, num_paths, T, device=device)
    timesteps = scheduler._get_inference_timesteps(device)
    keep = set(np.linspace(0, len(timesteps) - 1, 6, dtype=int).tolist())
    snapshots = []
    for i, t in enumerate(timesteps):
        t_batch = t.unsqueeze(0).expand(num_paths)
        eps = model.unet(x_t.reshape(num_paths, 1, T), t_batch, B_agent.reshape(num_paths, -1))
        x_t = scheduler._ddim_step(x_t, eps.reshape(1, num_paths, T), int(t.item()), i < len(timesteps) - 1)
        if i in keep:
            snapshots.append((int(t.item()), x_t[0, 0].detach().cpu().numpy().copy()))
    return x_t / model.config.target_scale, snapshots


@torch.no_grad()
def build_dashboard(
    checkpoint: Path,
    output_dir: Path,
    split: str = "val",
    num_contexts: int = 4,
    num_paths: int = 128,
    seed: int = 1234,
    device: str | None = None,
) -> dict:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = _load_model(checkpoint, device_t)
    ds = PathDataset(model.config, split=split)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(ds), size=min(num_contexts, len(ds)), replace=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_real = []
    all_gen = []
    records = []

    for panel_idx, ds_idx in enumerate(indices):
        sample = ds[int(ds_idx)]
        short = sample.short_seq.unsqueeze(0).to(device_t)
        mid = sample.mid_seq.unsqueeze(0).to(device_t)
        long = sample.long_seq.unsqueeze(0).to(device_t)
        target = sample.target.numpy() / model.config.target_scale
        paths_t, snapshots = _sample_with_snapshots(model, short, mid, long, num_paths, seed + panel_idx)
        paths = paths_t[0].cpu().numpy()

        all_real.append(target)
        all_gen.append(paths)

        sq_err = ((paths - target[None, :]) ** 2).mean(axis=-1)
        best = int(sq_err.argmin())
        diversity = float(np.mean(np.std(paths, axis=0)))
        latent_sensitivity = float(np.mean(np.linalg.norm(paths - paths.mean(axis=0, keepdims=True), axis=1)))

        records.append({
            "sample_idx": int(sample.sample_idx),
            "best_path_mse": float(sq_err[best]),
            "mean_path_mse": float(sq_err.mean()),
            "path_diversity_mean_step_std": diversity,
            "latent_sensitivity_l2": latent_sensitivity,
            "real_turning_frequency": _turning_frequency(target),
            "generated_turning_frequency": _turning_frequency(paths),
            "real_max_drawdown": float(_max_drawdown(target[None, :])[0]),
            "generated_max_drawdown_mean": float(_max_drawdown(paths).mean()),
        })

        fig, axes = plt.subplots(2, 3, figsize=(18, 9))
        for p in paths:
            axes[0, 0].plot(p, color="steelblue", alpha=0.10, linewidth=0.5)
        axes[0, 0].plot(target, color="red", linewidth=2.0, label="real")
        axes[0, 0].plot(paths[best], color="green", linestyle="--", linewidth=1.5, label="best")
        axes[0, 0].set_title("Generated paths vs real")
        axes[0, 0].legend(fontsize=8)

        axes[0, 1].plot(paths[: min(16, len(paths))].T, alpha=0.65, linewidth=0.8)
        axes[0, 1].set_title("Latent diversity, same context")

        axes[0, 2].plot(np.abs(target), color="red", linewidth=2, label="real abs returns")
        axes[0, 2].plot(np.abs(paths).mean(axis=0), color="steelblue", label="gen mean abs returns")
        axes[0, 2].fill_between(
            np.arange(paths.shape[-1]),
            np.percentile(np.abs(paths), 10, axis=0),
            np.percentile(np.abs(paths), 90, axis=0),
            color="steelblue",
            alpha=0.2,
        )
        axes[0, 2].set_title("Volatility clustering proxy")
        axes[0, 2].legend(fontsize=8)

        axes[1, 0].hist(paths.reshape(-1), bins=70, alpha=0.7, density=True, label="generated")
        axes[1, 0].hist(target, bins=20, alpha=0.7, density=True, label="real")
        axes[1, 0].set_title("Return distribution")
        axes[1, 0].legend(fontsize=8)

        for t_val, snap in snapshots:
            axes[1, 1].plot(snap, linewidth=1.0, label=f"t={t_val}")
        axes[1, 1].set_title("DDIM denoising trajectory")
        axes[1, 1].legend(fontsize=7)

        axes[1, 2].plot(_lag_autocorr(target), color="red", label="real")
        axes[1, 2].plot(_lag_autocorr(paths), color="steelblue", label="generated")
        axes[1, 2].axhline(0, color="gray", linewidth=0.5)
        axes[1, 2].set_title("Autocorrelation decay")
        axes[1, 2].legend(fontsize=8)

        fig.suptitle(f"{checkpoint.name} | sample_idx={sample.sample_idx}")
        fig.tight_layout()
        fig.savefig(output_dir / f"emergence_panel_{panel_idx:02d}.png", dpi=130)
        plt.close(fig)

    real_arr = np.stack(all_real)
    gen_arr = np.concatenate(all_gen, axis=0)
    summary = {
        "checkpoint": str(checkpoint),
        "split": split,
        "num_contexts": int(len(indices)),
        "num_paths": int(num_paths),
        "real_moments": _moments(real_arr),
        "generated_moments": _moments(gen_arr),
        "real_autocorr_lags_1_10": _lag_autocorr(real_arr).tolist(),
        "generated_autocorr_lags_1_10": _lag_autocorr(gen_arr).tolist(),
        "records": records,
    }
    with open(output_dir / "emergence_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build emergence dashboard panels for a checkpoint")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("checkpoints/emergence_dashboard"), type=Path)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--num-contexts", default=4, type=int)
    parser.add_argument("--num-paths", default=128, type=int)
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    summary = build_dashboard(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        split=args.split,
        num_contexts=args.num_contexts,
        num_paths=args.num_paths,
        seed=args.seed,
        device=args.device,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
