"""Training loop for MMFPS_GEN_V2 behavioral diffusion generator.

Features:
  - Mixed precision (torch.cuda.amp)
  - Gradient accumulation
  - EMA weight shadow
  - Cosine LR schedule with linear warmup
  - Checkpoint save / resume
  - Per-step metric logging + periodic visualization
  - Safety monitoring (NaN/Inf detection, gradient explosion skip)
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import BehaviorGenConfig
from dataset import PathDataset, build_splits, collate_fn
from generator import BehaviorDiffusionGenerator
from losses import compute_all_losses
from metrics_tracker import MetricsTracker
from safety import TrainingSafetyMonitor, check_gradients, compute_gradient_health


class EMAWrapper:
    """Exponential Moving Average of model weights.

    Usage:
        ema = EMAWrapper(model, decay=0.9999)
        ema.update()  # call after each optimizer step
        ema.apply()   # swap EMA weights into model for eval
        ema.restore() # swap original weights back
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        self.backup: dict[str, torch.Tensor] = {}
        self._registered = False

    def _register(self) -> None:
        if self._registered:
            return
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()
        self._registered = True

    def update(self) -> None:
        self._register()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply(self) -> None:
        """Swap EMA parameters into model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self) -> None:
        """Restore original parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup.clear()


def _build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    config: BehaviorGenConfig,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine schedule with linear warmup."""
    warmup = config.warmup_steps
    total = max(total_steps, warmup + 1)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(total - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _save_checkpoint(
    model: BehaviorDiffusionGenerator,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    ema: EMAWrapper,
    step: int,
    epoch: int,
    metrics: dict[str, float],
    output_dir: Path,
    tag: str = "",
) -> Path:
    """Save training state to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"step_{step}" if not tag else f"step_{step}_{tag}"
    path = output_dir / f"{filename}.pt"

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "ema_shadow": ema.shadow,
        "step": step,
        "epoch": epoch,
        "metrics": metrics,
        "config": model.config,
    }, path)

    return path


def _load_checkpoint(
    path: Path,
    model: BehaviorDiffusionGenerator,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    ema: EMAWrapper,
    device: torch.device,
) -> int:
    """Load training state. Returns global step."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if ckpt.get("ema_shadow"):
        ema.shadow = ckpt["ema_shadow"]
        ema._registered = True
    return ckpt.get("step", 0)


def train(
    output_dir: str,
    config: BehaviorGenConfig | None = None,
    resume_from: Optional[str] = None,
    device: Optional[str] = None,
    ema_enabled: bool = True,
) -> BehaviorDiffusionGenerator:
    """Run full training pipeline.

    Args:
        output_dir: Directory for checkpoints and logs
        config: Hyperparameter config (uses defaults if None)
        resume_from: Path to checkpoint .pt file to resume
        device: "cuda" or "cpu" (auto-detect if None)

    Returns:
        Trained generator model (with EMA weights applied)
    """
    cfg = config or BehaviorGenConfig()
    total_steps = getattr(cfg, '_total_steps', None) or 0
    max_epochs = getattr(cfg, '_max_epochs', None) or 10

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ───────────────────────────────────────────────────────
    train_ds, val_ds, test_ds = build_splits(cfg)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device == "cuda"),
        drop_last=True,
        collate_fn=collate_fn,
    )

    print("num_workers =", train_loader.num_workers)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    print(f"Batches/epoch: {len(train_loader)}")

    # ── Model ──────────────────────────────────────────────────────
    model = BehaviorDiffusionGenerator(cfg).to(device_t)
    params_m = model.count_parameters() / 1e6
    print(f"Model: {params_m:.1f}M params")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    if total_steps <= 0:
        total_steps = len(train_loader) * 10  # ~10 epochs default
    scheduler = _build_lr_scheduler(optimizer, cfg, total_steps)
    ema = EMAWrapper(model, decay=cfg.ema_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    tracker = MetricsTracker()

    # ── Safety monitor ─────────────────────────────────────────
    safety_monitor = TrainingSafetyMonitor(ema_decay=0.95)

    # ── Resume ─────────────────────────────────────────────────────
    global_step = 0
    if resume_from and Path(resume_from).exists():
        global_step = _load_checkpoint(
            Path(resume_from), model, optimizer, scheduler, ema, device_t
        )
        print(f"Resumed from {resume_from} at step {global_step}")

    # ── Logging ────────────────────────────────────────────────────
    log_path = output_dir / "training_log.jsonl"
    buf = []

    def log_entry(data: dict) -> None:
        entry = {"timestamp": datetime.utcnow().isoformat(), **data}
        buf.append(json.dumps(entry) + "\n")

    def flush_log() -> None:
        if buf:
            with open(log_path, "a") as f:
                f.writelines(buf)
            buf.clear()

    # ── Training loop ───────────────────────────────────────────────
    K = cfg.training_paths_per_sample
    accumulation_counter = 0
    unclipped_norm = 0.0
    grad_health = {}

    model.train()

    pbar = tqdm(total=total_steps, desc="Training")

    train_iter = iter(train_loader)

    while global_step < total_steps:

        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        short = batch.short_seq.to(device_t)
        mid = batch.mid_seq.to(device_t)
        long = batch.long_seq.to(device_t)
        target = batch.target.to(device_t)

        B = short.shape[0]
        T = target.shape[1]

        with torch.amp.autocast("cuda", enabled=False):
            B0 = model.base_encoder(short, mid, long)
            B_agent, z = model.agent_module(B0, K)

            clean = target.unsqueeze(1).expand(-1, K, -1)
            clean_flat = clean.reshape(B * K, 1, -1)
            beh_flat = B_agent.reshape(B * K, -1)

            t = torch.randint(
                0, cfg.diffusion_timesteps,
                (B * K,), device=device_t, dtype=torch.long
            )

            noisy, noise_true = model.scheduler.add_noise(clean_flat, t)
            noise_pred = model(noisy, t, beh_flat)

            structural_weight = (
                cfg.weight_volatility + cfg.weight_trend + cfg.weight_turning
                + cfg.weight_diversity + cfg.weight_latent_sensitivity
                + cfg.weight_manifold_spread
                + getattr(cfg, "weight_smoothness", 0.0)
                + getattr(cfg, "weight_autocorr", 0.0)
                + getattr(cfg, "weight_tail", 0.0)
            )
            x0_context = torch.enable_grad() if structural_weight > 0 else torch.no_grad()
            with x0_context:
                sqrt_ac = model.scheduler.sqrt_alpha_cumprod.to(device_t)
                sqrt_1mac = model.scheduler.sqrt_one_minus_alpha_cumprod.to(device_t)
                x0_pred_flat = (
                    noisy - sqrt_1mac[t].reshape(-1, 1, 1) * noise_pred
                ) / sqrt_ac[t].reshape(-1, 1, 1).clamp(min=1e-4)
                x0_pred = x0_pred_flat.reshape(B, K, T)

            losses = compute_all_losses(noise_pred, noise_true, x0_pred, target, B_agent, cfg)

        # ── NaN / Inf guard ─────────────────────────────────────────
        loss = losses["total"] / cfg.accumulation_steps

        if not torch.isfinite(loss):
            print(f"\n[SAFETY] Non-finite total loss at step {global_step}. Skipping.")
            optimizer.zero_grad(set_to_none=True)
            accumulation_counter = 0
            safety_monitor.record_skip()
            global_step += 1
            pbar.update(1)
            continue

        # Backward
        scaler.scale(loss).backward()
        accumulation_counter += 1

        if accumulation_counter >= cfg.accumulation_steps:
            scaler.unscale_(optimizer)

            # ── Gradient health analysis (read-only, no modification) ────────────
            grad_health = compute_gradient_health(model)
            
            # Check for NaN/Inf BEFORE clipping
            grads_ok, grad_stats = check_gradients(model)

            # Compute unclipped norm
            unclipped_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    unclipped_norm += float(p.grad.norm().square())
            unclipped_norm = math.sqrt(unclipped_norm)

            # Safety decision: skip if gradients are pathological
            should_skip = (
                not grads_ok or
                unclipped_norm > 1000.0 or  # Massive explosion (loose threshold for 11M params)
                unclipped_norm < 1e-8       # Vanishing
            )

            safety_monitor.record_batch(
                loss, float(unclipped_norm),
                nan=grad_stats["grad_has_nan"] > 0,
                inf=grad_stats["grad_has_inf"] > 0,
            )

            if should_skip:
                print(
                    f"\n[SAFETY] Pathological gradients at step {global_step}. "
                    f"unclipped_norm={unclipped_norm:.2f}, "
                    f"has_nan={grad_stats['grad_has_nan']}, "
                    f"has_inf={grad_stats['grad_has_inf']}. Skipping optimizer step."
                )
                optimizer.zero_grad(set_to_none=True)
                scaler.step(optimizer)  # reset scaler unscale flag
                scaler.update()
                accumulation_counter = 0
                safety_monitor.record_skip()
                global_step += 1
                pbar.update(1)
                continue

            # If gradients are OK, apply conservative clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            grad_norm_clipped = sum(p.grad.norm().square() for p in model.parameters() if p.grad is not None) ** 0.5

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            if ema_enabled:
                ema.update()
            accumulation_counter = 0

        # ── Tracking (cheap diagnostics only) ────────────────
        step_metrics = {f"loss/{k}": float(v.item()) for k, v in losses.items()}
        with torch.no_grad():
            step_metrics["diag/x0_mean"] = float(x0_pred.mean().item())
            step_metrics["diag/x0_std"] = float(x0_pred.std().item())
            noise_mse = (noise_pred - noise_true).pow(2).mean()
            step_metrics["diag/noise_mse"] = float(noise_mse.item())
            alpha_t = model.scheduler.alphas_cumprod.to(device_t)[t]
            step_metrics["diag/snr"] = float((alpha_t / (1.0 - alpha_t + 1e-8)).mean().item())

            manifold_m = MetricsTracker.compute_manifold_metrics(x0_pred, B_agent)
            structural_m = MetricsTracker.compute_structural_metrics(x0_pred, target)
            step_metrics.update(manifold_m)
            step_metrics.update(structural_m)

        step_metrics["train/lr"] = scheduler.get_last_lr()[0]
        if accumulation_counter == 0:
            step_metrics["train/grad_norm"] = float(unclipped_norm)
            for key, val in grad_health.items():
                if isinstance(val, (int, float)):
                    step_metrics[f"grad_health/{key}"] = float(val)

        safety = safety_monitor.summary()
        step_metrics["safety/skip_rate"] = safety["skip_rate"]
        step_metrics["safety/ema_grad_norm"] = safety["ema_grad_norm"]

        tracker.update(step_metrics)

        # ── Logging ────────────────────────────────────────
        global_step += 1
        pbar.update(1)

        if global_step % cfg.log_every == 0:
            pbar.set_postfix({
                "L": f"{step_metrics.get('loss/total', 0):.3f}",
                "rec": f"{step_metrics.get('loss/reconstruction', 0):.3f}",
                "x0s": f"{step_metrics.get('diag/x0_std', 0):.4f}",
                "nmse": f"{step_metrics.get('diag/noise_mse', 0):.4f}",
                "gn": f"{step_metrics.get('train/grad_norm', 0):.2f}",
                "cov": f"{structural_m.get('structural/cone_coverage', 0):.2%}",
            })
            log_entry({**step_metrics, "step": global_step, "epoch": 0})

        # ── Checkpoint ─────────────────────────────────────
        if global_step % cfg.checkpoint_every == 0:
            flush_log()
            path = _save_checkpoint(
                model, optimizer, scheduler, ema,
                global_step, 0, tracker.summary(), output_dir
            )
            print(f"\nCheckpoint: {path}")

        # ── Visualization ───────────────────────────────────
        if global_step % cfg.visualize_every == 0:
            _viz_callback(model, val_ds, output_dir / "viz", global_step, device_t)

    # ── Final save ──────────────────────────────────────────────────
    if ema_enabled:
        ema.apply()
    flush_log()
    final_path = _save_checkpoint(
        model, optimizer, scheduler, ema,
        global_step, 0, tracker.summary(), output_dir, tag="final"
    )
    print(f"Final checkpoint: {final_path}")

    return model


def _viz_callback(
    model: BehaviorDiffusionGenerator,
    val_ds: PathDataset,
    viz_dir: Path,
    step: int,
    device: torch.device,
    num_samples: int = 4,
) -> None:
    """Periodic visualization during training. Saves denoising evolution
    plots (PRIORITY) plus multi-path overlays and correlation heatmaps."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    viz_dir = viz_dir / f"step_{step:06d}"
    viz_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    model.scheduler.noise_scale_val = 1.0  # unit-variance initial noise
    indices = np.random.choice(
        len(val_ds), size=min(num_samples, len(val_ds)), replace=False
    )

    scheduler = model.scheduler

    with torch.no_grad():
        target_scale = model.config.target_scale
        for i, idx in enumerate(indices):
            sample = val_ds[idx]
            short = sample.short_seq.unsqueeze(0).to(device)
            mid   = sample.mid_seq.unsqueeze(0).to(device)
            long  = sample.long_seq.unsqueeze(0).to(device)
            target_scaled = sample.target.numpy()
            target = target_scaled / target_scale  # convert to raw return space

            B0 = model.base_encoder(short, mid, long)
            B_agent, _ = model.agent_module(B0, num_paths=32)

            # ── DENOISING EVOLUTION (PRIORITY) ───────────────────
            _viz_denoising_evolution(
                model, B_agent[:, :1, :], short, mid, long,
                viz_dir / f"denoise_s{i:02d}.png", device, target_scaled
            )

            # ── Multi-path overlay ────────────────────────────────
            paths_all, _, _ = model.generate(short, mid, long, num_paths=32)
            paths_np = paths_all[0].cpu().numpy()
            sq_err = ((paths_np - target) ** 2).mean(axis=-1)
            best_idx = sq_err.argmin()

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

            # Left: overlay
            for p in paths_np:
                ax1.plot(p, alpha=0.12, color="steelblue", linewidth=0.4)
            ax1.plot(target, "r-", linewidth=2, label="Real")
            ax1.plot(paths_np[best_idx], "g--", linewidth=1.5, label=f"Best idx={best_idx}")
            ax1.axhline(y=0, color="gray", linestyle=":", alpha=0.3)
            ax1.set_title(f"Step {step} — {len(paths_np)} Paths")
            ax1.legend(fontsize=8)

            # Right: correlation heatmap (top 50 paths)
            path_subset = paths_np[:min(50, len(paths_np))]
            corr = np.corrcoef(path_subset)
            im = ax2.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
            ax2.set_title("Path Correlation (anti-collapse check)")
            plt.colorbar(im, ax=ax2)

            fig.tight_layout()
            fig.savefig(viz_dir / f"sample_{i:02d}.png", dpi=100)
            plt.close(fig)

            # ── Endpoint histogram ────────────────────────────────
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(paths_np[:, -1], bins=30, color="steelblue", alpha=0.7, edgecolor="white")
            ax.axvline(x=target[-1], color="red", linewidth=2, label="Real final")
            ax.axvline(x=0, color="gray", linestyle=":", alpha=0.3)
            ax.set_title(f"Step {step} — Endpoint Distribution")
            ax.legend()
            fig.savefig(viz_dir / f"endpoints_{i:02d}.png", dpi=100)
            plt.close(fig)

    model.train()


def _viz_denoising_evolution(
    model: BehaviorDiffusionGenerator,
    B_agent_single: Tensor,      # (1, 1, 896) — single behavior for one path
    short: Tensor, mid: Tensor, long: Tensor,
    save_path: Path,
    device: torch.device,
    target: np.ndarray | None = None,
    num_snapshots: int = 7,
) -> None:
    """Visualize the DDIM denoising progression for ONE trajectory.

    This is the PRIMARY diagnostic for verifying true diffusion refinement.
    Saves a grid of subplots showing t=T → t=0 progression.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scheduler = model.scheduler
    T = model.config.path_horizon
    timesteps = scheduler._get_inference_timesteps(device)

    # Pick evenly spaced snapshots
    n_ts = len(timesteps)
    snap_indices = [
        n_ts - 1,
        n_ts - n_ts // 6,
        n_ts - 2 * n_ts // 6,
        n_ts - 3 * n_ts // 6,
        n_ts - 4 * n_ts // 6,
        n_ts - 5 * n_ts // 6,
        0,
    ]

    x_t = torch.randn(1, 1, T, device=device)
    beh_flat = B_agent_single.reshape(1, -1)

    snapshots = []

    for i, t_val in enumerate(timesteps):
        t = t_val.item()
        t_batch = t_val.unsqueeze(0).expand(1).to(device)
        x_flat = x_t.reshape(1, 1, T)
        eps_pred = model.unet(x_flat, t_batch, beh_flat).reshape(1, 1, T)
        x_t = scheduler._ddim_step(x_t, eps_pred, t, i < n_ts - 1)

        if i in snap_indices:
            snapshots.append((t_val.item(), x_t[0, 0].cpu().numpy().copy()))

    cols = min(4, len(snapshots))
    rows = (len(snapshots) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = np.atleast_1d(axes).flatten()

    for ax, (t_val, path) in zip(axes, snapshots):
        ax.plot(path, "b-", linewidth=1.5)
        if target is not None:
            ax.plot(target, "r-", linewidth=1, alpha=0.5, label="Real")
        ax.axhline(y=0, color="gray", linestyle=":", alpha=0.3)
        ax.set_title(f"t={t_val}", fontsize=9)
        ax.set_ylim(-4.0, 4.0)

    for j in range(len(snapshots), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("DDIM Denoising Evolution (noise → structure)", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
