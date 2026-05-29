"""Checkpoint-driven emergence snapshots and local HTML dashboard.

This module is intentionally observational. It does not change training losses,
the model architecture, or the sampler. It freezes a small validation suite and
128 per-path latent/noise identities, then records how those same stochastic
branches evolve across checkpoints.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import Tensor

from config import BehaviorGenConfig
from dataset import PathDataset, Sample
from generator import BehaviorDiffusionGenerator


SNAPSHOT_VERSION = 1
DEFAULT_SELECTED_DENOISE_PATHS = (0, 1, 2, 3)
SCENARIO_LABELS = (
    "trend_continuation",
    "trend_reversal",
    "breakout",
    "volatility_compression",
    "high_vol_regime",
    "range_bound",
    "post_gap_stabilization",
    "mean_reversion",
)


def _json_float(value: float | np.floating) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(value)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or b.size < 2:
        return 0.0
    if float(a.std()) < 1e-12 or float(b.std()) < 1e-12:
        return 0.0
    return _json_float(np.corrcoef(a, b)[0, 1])


def _kurtosis(x: np.ndarray) -> float:
    flat = np.asarray(x, dtype=np.float64).reshape(-1)
    std = flat.std()
    if std < 1e-12:
        return 0.0
    z = (flat - flat.mean()) / std
    return _json_float(np.mean(z**4))


def _lag1_autocorr(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    vals = []
    for row in arr:
        if row.size < 2:
            continue
        vals.append(_safe_corr(row[:-1], row[1:]))
    return _json_float(np.mean(vals)) if vals else 0.0


def _vol_cluster_lag1(x: np.ndarray) -> float:
    return _lag1_autocorr(np.abs(x))


def _turning_frequency(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[-1] < 3:
        return np.zeros(arr.shape[0], dtype=np.float64)
    signs = np.sign(np.diff(arr, axis=-1))
    turns = signs[:, 1:] * signs[:, :-1] < 0
    return turns.mean(axis=-1)


def _sign_match(path: np.ndarray, target: np.ndarray) -> float:
    p = np.sign(np.cumsum(path)[-1])
    t = np.sign(np.cumsum(target)[-1])
    return float(p == t)


def _path_metrics(paths: np.ndarray, target: np.ndarray) -> list[dict[str, float]]:
    target = np.asarray(target, dtype=np.float64)
    target_vol = float(np.std(target) + 1e-12)
    target_total = float(np.sum(target))
    target_turn = float(_turning_frequency(target)[0])
    rows = []
    for idx, path in enumerate(np.asarray(paths, dtype=np.float64)):
        mse = float(np.mean((path - target) ** 2))
        rows.append({
            "path_index": int(idx),
            "mse": _json_float(mse),
            "rmse": _json_float(math.sqrt(max(mse, 0.0))),
            "mae": _json_float(np.mean(np.abs(path - target))),
            "corr": _safe_corr(path, target),
            "direction_match": _sign_match(path, target),
            "magnitude_error": _json_float(abs(float(path.sum()) - target_total)),
            "volatility_ratio": _json_float(float(np.std(path) + 1e-12) / target_vol),
            "turning_error": _json_float(abs(float(_turning_frequency(path)[0]) - target_turn)),
        })
    return rows


def _summary_metrics(paths: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    per_path = _path_metrics(paths, target)
    mses = np.array([row["mse"] for row in per_path], dtype=np.float64)
    best_idx = int(mses.argmin())
    corr = np.corrcoef(paths) if len(paths) > 1 else np.ones((1, 1))
    upper = corr[np.triu_indices_from(corr, k=1)] if corr.size else np.array([0.0])
    target_std = float(np.std(target) + 1e-12)
    return {
        "best_path_index": best_idx,
        "best_path_mse": _json_float(mses[best_idx]),
        "median_path_mse": _json_float(np.median(mses)),
        "mean_path_mse": _json_float(mses.mean()),
        "path_spread_mean_step_std": _json_float(np.std(paths, axis=0).mean()),
        "endpoint_spread": _json_float(np.std(np.cumsum(paths, axis=-1)[:, -1])),
        "median_variance_ratio": _json_float(np.median(np.var(paths, axis=-1) / (target_std**2))),
        "mean_abs_path_corr": _json_float(np.mean(np.abs(upper))) if upper.size else 0.0,
        "direction_coverage": _json_float(float(np.mean([row["direction_match"] for row in per_path]))),
        "generated_std": _json_float(np.std(paths)),
        "real_std": _json_float(np.std(target)),
        "generated_kurtosis": _kurtosis(paths),
        "real_kurtosis": _kurtosis(target),
        "generated_autocorr_lag1": _lag1_autocorr(paths),
        "real_autocorr_lag1": _lag1_autocorr(target),
        "generated_vol_cluster_lag1": _vol_cluster_lag1(paths),
        "real_vol_cluster_lag1": _vol_cluster_lag1(target),
        "generated_turning_frequency": _json_float(_turning_frequency(paths).mean()),
        "real_turning_frequency": _json_float(_turning_frequency(target)[0]),
    }


def _sample_score(sample: Sample) -> dict[str, float]:
    context_proxy = sample.long_seq[:, 0].numpy().astype(np.float64)
    target = sample.target.numpy().astype(np.float64)
    recent = context_proxy[-min(20, len(context_proxy)):]
    earlier = context_proxy[:max(1, min(20, len(context_proxy)))]
    context_slope = float(recent.mean() - earlier.mean())
    future_sum = float(target.sum())
    future_vol = float(target.std())
    context_vol = float(context_proxy.std())
    zero_cross = float(np.mean(np.sign(target[1:]) != np.sign(target[:-1]))) if len(target) > 1 else 0.0
    first_abs = float(abs(target[0])) if len(target) else 0.0
    return {
        "context_slope": context_slope,
        "future_sum": future_sum,
        "future_vol": future_vol,
        "context_vol": context_vol,
        "zero_cross": zero_cross,
        "first_abs": first_abs,
        "same_direction": float(np.sign(context_slope) == np.sign(future_sum)),
    }


def _choose_scenarios(ds: PathDataset, num_scenarios: int, seed: int, scan: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    scan_n = min(scan, len(ds))
    candidate_indices = rng.choice(len(ds), size=scan_n, replace=False)
    rows = []
    for idx in candidate_indices:
        sample = ds[int(idx)]
        score = _sample_score(sample)
        rows.append({"dataset_index": int(idx), "sample_idx": int(sample.sample_idx), **score})

    def pick(label: str, used: set[int]) -> dict | None:
        available = [row for row in rows if row["dataset_index"] not in used]
        if not available:
            return None
        if label == "trend_continuation":
            key = lambda r: abs(r["future_sum"]) + (1.0 if r["same_direction"] else -1.0)
        elif label == "trend_reversal":
            key = lambda r: abs(r["future_sum"]) + (0.0 if r["same_direction"] else 1.0)
        elif label == "breakout":
            key = lambda r: abs(r["future_sum"]) / (r["context_vol"] + 1e-6)
        elif label == "volatility_compression":
            key = lambda r: r["context_vol"] / (r["future_vol"] + 1e-6)
        elif label == "high_vol_regime":
            key = lambda r: r["context_vol"] + r["future_vol"]
        elif label == "range_bound":
            key = lambda r: r["zero_cross"] - abs(r["future_sum"])
        elif label == "post_gap_stabilization":
            key = lambda r: r["first_abs"] / (r["future_vol"] + 1e-6)
        elif label == "mean_reversion":
            key = lambda r: -abs(r["future_sum"]) + r["zero_cross"]
        else:
            key = lambda r: r["future_vol"]
        return max(available, key=key)

    selected = []
    used: set[int] = set()
    for label in SCENARIO_LABELS[:num_scenarios]:
        row = pick(label, used)
        if row is None:
            break
        used.add(row["dataset_index"])
        selected.append({**row, "scenario_id": label})

    while len(selected) < min(num_scenarios, len(rows)):
        row = rows[len(selected)]
        if row["dataset_index"] not in used:
            selected.append({**row, "scenario_id": f"scenario_{len(selected):02d}"})
            used.add(row["dataset_index"])
    return selected


def load_or_create_scenario_suite(
    output_root: Path,
    config: BehaviorGenConfig,
    split: str = "val",
    num_scenarios: int = 8,
    seed: int = 1234,
    scan: int = 512,
) -> list[dict]:
    suite_path = output_root / "scenario_suite.json"
    if suite_path.exists():
        with open(suite_path, "r", encoding="utf-8") as f:
            return json.load(f)["scenarios"]

    ds = PathDataset(config, split=split)
    scenarios = _choose_scenarios(ds, num_scenarios=num_scenarios, seed=seed, scan=scan)
    suite = {
        "version": SNAPSHOT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "seed": seed,
        "scenarios": scenarios,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    with open(suite_path, "w", encoding="utf-8") as f:
        json.dump(suite, f, indent=2)
    return scenarios


def _fixed_latents(
    batch: int,
    num_paths: int,
    latent_dim: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return torch.randn(batch, num_paths, latent_dim, generator=gen, device=device, dtype=dtype)


def _fixed_noise(
    batch: int,
    num_paths: int,
    horizon: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return torch.randn(batch, num_paths, horizon, generator=gen, device=device, dtype=dtype)


@torch.no_grad()
def generate_fixed_paths(
    model: BehaviorDiffusionGenerator,
    short: Tensor,
    mid: Tensor,
    long: Tensor,
    num_paths: int,
    latent_seed: int,
    noise_seed: int,
    denoise_path_indices: Iterable[int] = DEFAULT_SELECTED_DENOISE_PATHS,
    num_denoise_snapshots: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int], list[int]]:
    device = short.device
    dtype = short.dtype
    batch = short.shape[0]
    horizon = model.config.path_horizon
    timesteps = model.scheduler._get_inference_timesteps(device)
    keep_indices = set(np.linspace(0, len(timesteps) - 1, num_denoise_snapshots, dtype=int).tolist())
    selected = [idx for idx in denoise_path_indices if 0 <= idx < num_paths]

    b0 = model.base_encoder(short, mid, long)
    z = _fixed_latents(
        batch=batch,
        num_paths=num_paths,
        latent_dim=model.config.agent_behavior_dim,
        seed=latent_seed,
        device=device,
        dtype=dtype,
    )
    b_agent, z_used = model.agent_module(b0, num_paths, z=z)
    x_t = model.scheduler.noise_scale_val * _fixed_noise(
        batch=batch,
        num_paths=num_paths,
        horizon=horizon,
        seed=noise_seed,
        device=device,
        dtype=dtype,
    )

    snapshots = []
    snapshot_timesteps = []
    if 0 in keep_indices:
        snapshots.append(x_t[0, selected].detach().cpu().numpy() / model.config.target_scale)
        snapshot_timesteps.append(int(timesteps[0].item()))

    for step_idx, t in enumerate(timesteps):
        t_batch = t.unsqueeze(0).expand(batch * num_paths)
        eps = model.unet(
            x_t.reshape(batch * num_paths, 1, horizon),
            t_batch,
            b_agent.reshape(batch * num_paths, -1),
        )
        x_t = model.scheduler._ddim_step(
            x_t,
            eps.reshape(batch, num_paths, horizon),
            int(t.item()),
            step_idx < len(timesteps) - 1,
        )
        if step_idx in keep_indices:
            snapshots.append(x_t[0, selected].detach().cpu().numpy() / model.config.target_scale)
            snapshot_timesteps.append(int(t.item()))

    paths = x_t.detach().cpu().numpy() / model.config.target_scale
    return (
        paths,
        z_used.detach().cpu().numpy(),
        b_agent.detach().cpu().numpy(),
        np.asarray(snapshots, dtype=np.float32),
        snapshot_timesteps,
        selected,
    )


def _load_model_from_checkpoint(checkpoint: Path, device: torch.device) -> BehaviorDiffusionGenerator:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("config", BehaviorGenConfig())
    model = BehaviorDiffusionGenerator(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.scheduler.noise_scale_val = 1.0
    return model


def _step_from_checkpoint(checkpoint: Path, fallback: int = 0) -> int:
    match = re.search(r"step_(\d+)", checkpoint.name)
    return int(match.group(1)) if match else fallback


@torch.no_grad()
def emit_emergence_snapshot(
    model: BehaviorDiffusionGenerator,
    val_ds: PathDataset,
    output_root: Path,
    step: int,
    checkpoint_path: Path | None = None,
    split: str = "val",
    num_scenarios: int = 8,
    num_paths: int = 128,
    seed: int = 1234,
    scenario_scan: int = 512,
    device: torch.device | None = None,
) -> Path:
    """Write one compact npz/json snapshot for live emergence replay."""
    if device is None:
        device = next(model.parameters()).device
    output_root.mkdir(parents=True, exist_ok=True)
    scenarios = load_or_create_scenario_suite(
        output_root=output_root,
        config=model.config,
        split=split,
        num_scenarios=num_scenarios,
        seed=seed,
        scan=scenario_scan,
    )

    was_training = model.training
    model.eval()
    model.scheduler.noise_scale_val = 1.0

    contexts = []
    targets = []
    paths_all = []
    z_all = []
    denoise_all = []
    per_scenario_metrics = []
    per_path_metrics = []
    denoise_timesteps: list[int] | None = None
    selected_denoise_paths: list[int] | None = None

    for scenario_idx, scenario in enumerate(scenarios[:num_scenarios]):
        sample = val_ds[int(scenario["dataset_index"])]
        short = sample.short_seq.unsqueeze(0).to(device)
        mid = sample.mid_seq.unsqueeze(0).to(device)
        long = sample.long_seq.unsqueeze(0).to(device)
        target = sample.target.numpy() / model.config.target_scale
        paths, z, _, denoise, timesteps, selected = generate_fixed_paths(
            model=model,
            short=short,
            mid=mid,
            long=long,
            num_paths=num_paths,
            latent_seed=seed + scenario_idx * 1000 + 17,
            noise_seed=seed + scenario_idx * 1000 + 29,
        )
        paths = paths[0]
        z = z[0]

        contexts.append(sample.long_seq[:, 0].numpy().astype(np.float32))
        targets.append(target.astype(np.float32))
        paths_all.append(paths.astype(np.float32))
        z_all.append(z.astype(np.float32))
        denoise_all.append(denoise.astype(np.float32))

        scenario_metrics = _summary_metrics(paths, target)
        scenario_metrics.update({
            "scenario_index": scenario_idx,
            "scenario_id": scenario["scenario_id"],
            "dataset_index": int(scenario["dataset_index"]),
            "sample_idx": int(sample.sample_idx),
        })
        per_scenario_metrics.append(scenario_metrics)

        path_rows = _path_metrics(paths, target)
        for row in path_rows:
            row.update({
                "scenario_index": scenario_idx,
                "scenario_id": scenario["scenario_id"],
            })
        per_path_metrics.append(path_rows)
        denoise_timesteps = timesteps
        selected_denoise_paths = selected

    snapshot_dir = output_root / f"step_{step:06d}"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    npz_path = snapshot_dir / "snapshot.npz"
    meta_path = snapshot_dir / "metrics.json"
    np.savez_compressed(
        npz_path,
        context=np.stack(contexts),
        real_future=np.stack(targets),
        generated_futures=np.stack(paths_all),
        latent_z=np.stack(z_all),
        denoising_states=np.stack(denoise_all),
    )

    meta = {
        "version": SNAPSHOT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "step": int(step),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "split": split,
        "num_scenarios": int(len(contexts)),
        "num_paths": int(num_paths),
        "path_horizon": int(model.config.path_horizon),
        "context_horizon": int(model.config.long_horizon),
        "target_scale": float(model.config.target_scale),
        "latent_seed_base": int(seed),
        "path_seed_labels": [int(seed + idx) for idx in range(num_paths)],
        "denoise_timesteps": denoise_timesteps or [],
        "selected_denoise_paths": selected_denoise_paths or [],
        "scenarios": scenarios[:num_scenarios],
        "scenario_metrics": per_scenario_metrics,
        "path_metrics": per_path_metrics,
        "config": asdict(model.config),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    rebuild_dashboard(output_root)
    if was_training:
        model.train()
    return snapshot_dir


def rebuild_dashboard(snapshot_root: Path, output_html: Path | None = None) -> Path:
    """Create/refresh a self-contained local HTML dashboard."""
    if output_html is None:
        output_html = snapshot_root / "dashboard.html"
    snapshot_dirs = sorted(p for p in snapshot_root.glob("step_*") if (p / "snapshot.npz").exists())
    snapshots = []
    arrays: dict[str, dict[str, list]] = {}
    for snap_dir in snapshot_dirs:
        meta_path = snap_dir / "metrics.json"
        if not meta_path.exists():
            continue
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        rel = snap_dir.name
        data = np.load(snap_dir / "snapshot.npz")
        arrays[rel] = {
            "context": np.asarray(data["context"]).round(8).tolist(),
            "real_future": np.asarray(data["real_future"]).round(8).tolist(),
            "generated_futures": np.asarray(data["generated_futures"]).round(8).tolist(),
            "denoising_states": np.asarray(data["denoising_states"]).round(8).tolist(),
        }
        snapshots.append({
            "dir": rel,
            "step": meta["step"],
            "created_at": meta["created_at"],
            "checkpoint_path": meta.get("checkpoint_path"),
            "scenarios": meta["scenarios"],
            "scenario_metrics": meta["scenario_metrics"],
            "path_metrics": meta["path_metrics"],
            "denoise_timesteps": meta.get("denoise_timesteps", []),
            "selected_denoise_paths": meta.get("selected_denoise_paths", []),
        })

    html = _dashboard_html({
        "snapshots": snapshots,
        "arrays": arrays,
    })
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html, encoding="utf-8")
    return output_html


def _dashboard_html(payload: dict) -> str:
    payload_json = json.dumps(payload)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MMFPS Emergence Dashboard</title>
  <style>
    :root {{
      --bg: #101316;
      --panel: #171c20;
      --panel2: #1d2429;
      --text: #e9eef2;
      --muted: #93a3ad;
      --line: #2b3840;
      --blue: #58a6ff;
      --green: #44d17b;
      --red: #ff6b6b;
      --yellow: #f2c94c;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Segoe UI, Roboto, Arial, sans-serif; background: var(--bg); color: var(--text); }}
    header {{ position: sticky; top: 0; z-index: 5; background: #0c0f12; border-bottom: 1px solid var(--line); padding: 12px 18px; }}
    h1 {{ margin: 0 0 10px; font-size: 20px; font-weight: 650; }}
    .controls {{ display: flex; gap: 10px; flex-wrap: wrap; align-items: end; }}
    label {{ display: grid; gap: 4px; font-size: 12px; color: var(--muted); }}
    select, input, button {{ background: var(--panel2); color: var(--text); border: 1px solid var(--line); border-radius: 6px; padding: 8px 10px; }}
    button {{ cursor: pointer; }}
    main {{ padding: 16px; display: grid; gap: 16px; }}
    .grid {{ display: grid; grid-template-columns: 1.35fr 0.65fr; gap: 16px; }}
    .panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 12px; min-width: 0; }}
    .metrics {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; }}
    .metric {{ background: var(--panel2); border: 1px solid var(--line); border-radius: 6px; padding: 9px; }}
    .metric span {{ display: block; font-size: 11px; color: var(--muted); }}
    .metric strong {{ font-size: 18px; }}
    canvas {{ width: 100%; background: #0b0e11; border: 1px solid var(--line); border-radius: 6px; }}
    #mainPlot {{ height: 470px; }}
    #denoisePlot {{ height: 300px; }}
    #detailPlot {{ height: 300px; }}
    .tiles {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); gap: 8px; max-height: 620px; overflow: auto; padding-right: 4px; }}
    .tile {{ background: var(--panel2); border: 1px solid var(--line); border-radius: 6px; padding: 6px; cursor: pointer; }}
    .tile.active {{ outline: 2px solid var(--yellow); }}
    .tile.best {{ border-color: var(--green); }}
    .tile canvas {{ height: 70px; }}
    .tile-label {{ display: flex; justify-content: space-between; gap: 4px; font-size: 11px; color: var(--muted); margin-bottom: 4px; }}
    .history {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 8px; }}
    .checkpoint-card {{ background: var(--panel2); border: 1px solid var(--line); border-radius: 6px; padding: 10px; }}
    .small {{ font-size: 12px; color: var(--muted); }}
    @media (max-width: 980px) {{ .grid {{ grid-template-columns: 1fr; }} .metrics {{ grid-template-columns: repeat(2, 1fr); }} }}
  </style>
</head>
<body>
  <header>
    <h1>MMFPS_GEN_V2 Emergence Dashboard</h1>
    <div class="controls">
      <label>Checkpoint<select id="checkpointSelect"></select></label>
      <label>Scenario<select id="scenarioSelect"></select></label>
      <label>Path<select id="pathSelect"></select></label>
      <label><span>Show all paths</span><input id="showAll" type="checkbox" checked></label>
      <button id="prevStep">Prev checkpoint</button>
      <button id="nextStep">Next checkpoint</button>
    </div>
  </header>
  <main>
    <section class="metrics" id="metrics"></section>
    <section class="grid">
      <div class="panel">
        <h2>Main Comparison</h2>
        <canvas id="mainPlot"></canvas>
      </div>
      <div class="panel">
        <h2>128 Independent Futures</h2>
        <div class="small">Each tile is a fixed latent/noise identity tracked across checkpoints.</div>
        <div class="tiles" id="tiles"></div>
      </div>
    </section>
    <section class="grid">
      <div class="panel">
        <h2>Selected Path Detail</h2>
        <canvas id="detailPlot"></canvas>
      </div>
      <div class="panel">
        <h2>Denoising Progression</h2>
        <canvas id="denoisePlot"></canvas>
      </div>
    </section>
    <section class="panel">
      <h2>Training Evolution</h2>
      <div class="history" id="history"></div>
    </section>
  </main>
  <script>
    const DATA = {payload_json};
    const state = {{ checkpoint: 0, scenario: 0, path: 0 }};
    const colors = {{ blue: '#58a6ff', green: '#44d17b', red: '#ff6b6b', yellow: '#f2c94c', muted: '#93a3ad' }};

    function byId(id) {{ return document.getElementById(id); }}
    function fmt(v, digits=4) {{
      if (v === undefined || v === null || Number.isNaN(v)) return 'n/a';
      if (Math.abs(v) < 0.001 && v !== 0) return Number(v).toExponential(2);
      return Number(v).toFixed(digits);
    }}
    function current() {{
      const snap = DATA.snapshots[state.checkpoint];
      const arr = DATA.arrays[snap.dir];
      return {{ snap, arr }};
    }}
    function seriesRange(seriesList) {{
      let min = Infinity, max = -Infinity;
      for (const s of seriesList) {{
        for (const v of s) {{ if (v < min) min = v; if (v > max) max = v; }}
      }}
      if (!Number.isFinite(min) || !Number.isFinite(max) || min === max) {{ min = -1; max = 1; }}
      const pad = (max - min) * 0.12 + 1e-9;
      return [min - pad, max + pad];
    }}
    function resizeCanvas(canvas) {{
      const rect = canvas.getBoundingClientRect();
      const scale = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.floor(rect.width * scale));
      canvas.height = Math.max(1, Math.floor(rect.height * scale));
      const ctx = canvas.getContext('2d');
      ctx.setTransform(scale, 0, 0, scale, 0, 0);
      return {{ ctx, w: rect.width, h: rect.height }};
    }}
    function drawLine(ctx, data, range, w, h, color, alpha=1, width=1.2, offsetX=0, scaleX=1) {{
      if (!data || data.length === 0) return;
      const [min, max] = range;
      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.beginPath();
      data.forEach((v, i) => {{
        const x = offsetX + (data.length === 1 ? 0 : i / (data.length - 1)) * scaleX;
        const y = h - ((v - min) / (max - min)) * h;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }});
      ctx.stroke();
      ctx.restore();
    }}
    function clearCanvas(canvas) {{
      const r = resizeCanvas(canvas);
      r.ctx.clearRect(0, 0, r.w, r.h);
      r.ctx.fillStyle = '#0b0e11';
      r.ctx.fillRect(0, 0, r.w, r.h);
      return r;
    }}
    function drawMain() {{
      const {{ snap, arr }} = current();
      const canvas = byId('mainPlot');
      const {{ ctx, w, h }} = clearCanvas(canvas);
      const real = arr.real_future[state.scenario];
      const paths = arr.generated_futures[state.scenario];
      const context = arr.context[state.scenario];
      const sm = snap.scenario_metrics[state.scenario];
      const best = paths[sm.best_path_index];
      const showAll = byId('showAll').checked;
      const all = showAll ? paths.concat([real, best]) : [real, best, paths[state.path]];
      const range = seriesRange(all);
      const ctxW = w * 0.36;
      const futW = w * 0.60;
      const gap = w * 0.04;
      const contextNorm = normalize(context, range);
      drawLine(ctx, contextNorm, range, w, h, colors.muted, 0.75, 1.0, 0, ctxW);
      if (showAll) for (const p of paths) drawLine(ctx, p, range, w, h, colors.blue, 0.09, 0.7, ctxW + gap, futW);
      drawLine(ctx, paths[state.path], range, w, h, colors.yellow, 0.95, 1.7, ctxW + gap, futW);
      drawLine(ctx, best, range, w, h, colors.green, 1, 2.2, ctxW + gap, futW);
      drawLine(ctx, real, range, w, h, colors.red, 1, 2.3, ctxW + gap, futW);
      ctx.strokeStyle = '#2b3840';
      ctx.beginPath(); ctx.moveTo(ctxW + gap / 2, 0); ctx.lineTo(ctxW + gap / 2, h); ctx.stroke();
      ctx.fillStyle = colors.muted; ctx.font = '12px Segoe UI';
      ctx.fillText('context feature proxy', 10, 18);
      ctx.fillStyle = colors.red; ctx.fillText('real future', ctxW + gap + 8, 18);
      ctx.fillStyle = colors.green; ctx.fillText('best generated', ctxW + gap + 96, 18);
      ctx.fillStyle = colors.yellow; ctx.fillText('selected path', ctxW + gap + 220, 18);
    }}
    function normalize(context, targetRange) {{
      const [tMin, tMax] = targetRange;
      const cMin = Math.min(...context), cMax = Math.max(...context);
      const mid = (tMin + tMax) / 2;
      const span = (tMax - tMin) * 0.35;
      return context.map(v => mid + ((v - cMin) / ((cMax - cMin) || 1) - 0.5) * span);
    }}
    function drawSimple(canvas, lines) {{
      const {{ ctx, w, h }} = clearCanvas(canvas);
      const range = seriesRange(lines.map(l => l.data));
      for (const l of lines) drawLine(ctx, l.data, range, w, h, l.color, l.alpha ?? 1, l.width ?? 1.2, 0, w);
    }}
    function drawDetail() {{
      const {{ snap, arr }} = current();
      const real = arr.real_future[state.scenario];
      const path = arr.generated_futures[state.scenario][state.path];
      const sm = snap.scenario_metrics[state.scenario];
      drawSimple(byId('detailPlot'), [
        {{ data: real, color: colors.red, width: 2.2 }},
        {{ data: path, color: colors.yellow, width: 2.0 }},
        {{ data: arr.generated_futures[state.scenario][sm.best_path_index], color: colors.green, width: 1.6, alpha: 0.9 }},
      ]);
    }}
    function drawDenoise() {{
      const {{ snap, arr }} = current();
      const den = arr.denoising_states[state.scenario];
      const selected = snap.selected_denoise_paths;
      let denoiseIdx = selected.indexOf(state.path);
      if (denoiseIdx < 0) denoiseIdx = 0;
      const lines = den.map((snapArr, i) => {{
        const opacity = 0.25 + 0.75 * (i / Math.max(1, den.length - 1));
        return {{ data: snapArr[denoiseIdx], color: i === den.length - 1 ? colors.green : colors.blue, alpha: opacity, width: i === den.length - 1 ? 2.2 : 1.0 }};
      }});
      lines.push({{ data: arr.real_future[state.scenario], color: colors.red, width: 1.6, alpha: 0.8 }});
      drawSimple(byId('denoisePlot'), lines);
    }}
    function renderMetrics() {{
      const {{ snap }} = current();
      const sm = snap.scenario_metrics[state.scenario];
      const pm = snap.path_metrics[state.scenario][state.path];
      const items = [
        ['step', snap.step, 0],
        ['best path', sm.best_path_index, 0],
        ['best MSE', sm.best_path_mse, 6],
        ['selected MSE', pm.mse, 6],
        ['direction coverage', sm.direction_coverage, 2],
        ['median var ratio', sm.median_variance_ratio, 2],
        ['vol cluster gen/real', `${{fmt(sm.generated_vol_cluster_lag1, 3)}} / ${{fmt(sm.real_vol_cluster_lag1, 3)}}`, null],
        ['kurtosis gen/real', `${{fmt(sm.generated_kurtosis, 2)}} / ${{fmt(sm.real_kurtosis, 2)}}`, null],
      ];
      byId('metrics').innerHTML = items.map(([label, value, digits]) =>
        `<div class="metric"><span>${{label}}</span><strong>${{digits === null ? value : fmt(value, digits)}}</strong></div>`
      ).join('');
    }}
    function renderTiles() {{
      const {{ snap, arr }} = current();
      const paths = arr.generated_futures[state.scenario];
      const sm = snap.scenario_metrics[state.scenario];
      const metrics = snap.path_metrics[state.scenario];
      const tiles = byId('tiles');
      tiles.innerHTML = '';
      paths.forEach((path, idx) => {{
        const div = document.createElement('div');
        div.className = `tile ${{idx === state.path ? 'active' : ''}} ${{idx === sm.best_path_index ? 'best' : ''}}`;
        div.innerHTML = `<div class="tile-label"><span>#${{idx}}</span><span>mse ${{fmt(metrics[idx].mse, 6)}}</span></div><canvas></canvas>`;
        div.onclick = () => {{ state.path = idx; syncSelectors(); renderAll(false); }};
        tiles.appendChild(div);
        drawSimple(div.querySelector('canvas'), [
          {{ data: arr.real_future[state.scenario], color: colors.red, width: 1.3, alpha: 0.65 }},
          {{ data: path, color: idx === sm.best_path_index ? colors.green : colors.blue, width: 1.2, alpha: 0.95 }},
        ]);
      }});
    }}
    function renderHistory() {{
      const cards = DATA.snapshots.map((snap, idx) => {{
        const sm = snap.scenario_metrics[state.scenario] || snap.scenario_metrics[0];
        return `<div class="checkpoint-card" onclick="state.checkpoint=${{idx}}; syncSelectors(); renderAll();">
          <strong>step ${{snap.step}}</strong>
          <div class="small">best path: ${{sm.best_path_index}}</div>
          <div class="small">best MSE: ${{fmt(sm.best_path_mse, 6)}}</div>
          <div class="small">spread: ${{fmt(sm.path_spread_mean_step_std, 6)}}</div>
          <div class="small">corr collapse: ${{fmt(sm.mean_abs_path_corr, 3)}}</div>
        </div>`;
      }});
      byId('history').innerHTML = cards.join('');
    }}
    function syncSelectors() {{
      const cur = current();
      const scenarioCount = cur.arr.generated_futures.length;
      const pathCount = cur.arr.generated_futures[Math.min(state.scenario, scenarioCount - 1)].length;
      if (state.scenario >= scenarioCount) state.scenario = 0;
      if (state.path >= pathCount) state.path = 0;
      byId('scenarioSelect').innerHTML = cur.snap.scenarios.map((s, i) => `<option value="${{i}}">${{s.scenario_id}}</option>`).join('');
      byId('pathSelect').innerHTML = Array.from({{ length: pathCount }}, (_, i) => `<option value="${{i}}">path ${{i}}</option>`).join('');
      byId('checkpointSelect').value = String(state.checkpoint);
      byId('scenarioSelect').value = String(state.scenario);
      byId('pathSelect').value = String(state.path);
    }}
    function setupSelectors() {{
      byId('checkpointSelect').innerHTML = DATA.snapshots.map((s, i) => `<option value="${{i}}">step ${{s.step}}</option>`).join('');
      byId('checkpointSelect').onchange = e => {{ state.checkpoint = Number(e.target.value); renderAll(); }};
      byId('scenarioSelect').onchange = e => {{ state.scenario = Number(e.target.value); renderAll(); }};
      byId('pathSelect').onchange = e => {{ state.path = Number(e.target.value); renderAll(); }};
      byId('showAll').onchange = () => renderAll(false);
      byId('prevStep').onclick = () => {{ state.checkpoint = Math.max(0, state.checkpoint - 1); syncSelectors(); renderAll(); }};
      byId('nextStep').onclick = () => {{ state.checkpoint = Math.min(DATA.snapshots.length - 1, state.checkpoint + 1); syncSelectors(); renderAll(); }};
    }}
    function renderAll(redrawTiles=true) {{
      if (!DATA.snapshots.length) return;
      syncSelectors();
      renderMetrics();
      drawMain();
      drawDetail();
      drawDenoise();
      if (redrawTiles) renderTiles();
      renderHistory();
    }}
    window.addEventListener('resize', () => renderAll(false));
    setupSelectors();
    renderAll();
    if (new URLSearchParams(window.location.search).get('live') === '1') {{
      setTimeout(() => window.location.reload(), 15000);
    }}
  </script>
</body>
</html>
"""


def emit_from_checkpoint(
    checkpoint: Path,
    output_root: Path,
    split: str,
    num_scenarios: int,
    num_paths: int,
    seed: int,
    device_name: str | None,
) -> Path:
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = _load_model_from_checkpoint(checkpoint, device)
    ds = PathDataset(model.config, split=split)
    return emit_emergence_snapshot(
        model=model,
        val_ds=ds,
        output_root=output_root,
        step=_step_from_checkpoint(checkpoint),
        checkpoint_path=checkpoint,
        split=split,
        num_scenarios=num_scenarios,
        num_paths=num_paths,
        seed=seed,
        device=device,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Emit fixed-latent emergence snapshots and dashboard.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    emit = sub.add_parser("emit", help="Generate a snapshot from one checkpoint.")
    emit.add_argument("--checkpoint", type=Path, required=True)
    emit.add_argument("--output-root", type=Path, default=Path("emergence_snapshots"))
    emit.add_argument("--split", choices=["train", "val", "test"], default="val")
    emit.add_argument("--num-scenarios", type=int, default=8)
    emit.add_argument("--num-paths", type=int, default=128)
    emit.add_argument("--seed", type=int, default=1234)
    emit.add_argument("--device", default=None)

    build = sub.add_parser("build-dashboard", help="Rebuild dashboard.html from existing snapshots.")
    build.add_argument("--snapshot-root", type=Path, default=Path("emergence_snapshots"))
    build.add_argument("--output-html", type=Path, default=None)

    args = parser.parse_args()
    if args.cmd == "emit":
        snap = emit_from_checkpoint(
            checkpoint=args.checkpoint,
            output_root=args.output_root,
            split=args.split,
            num_scenarios=args.num_scenarios,
            num_paths=args.num_paths,
            seed=args.seed,
            device_name=args.device,
        )
        print(f"Snapshot saved to {snap}")
        print(f"Dashboard: {args.output_root / 'dashboard.html'}")
    elif args.cmd == "build-dashboard":
        html = rebuild_dashboard(args.snapshot_root, args.output_html)
        print(f"Dashboard: {html}")


if __name__ == "__main__":
    main()
