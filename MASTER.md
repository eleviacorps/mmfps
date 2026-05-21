# MMFPS_GEN_V2 Master Journal

This journal records the current validation stance, the important edits made to
the codebase, and why each edit exists. The project is now in controlled
emergence-validation mode: no major architectural rewrites should happen until
pure diffusion behavior has been observed and measured.

## Current Research Posture

Priority from this point:

1. Validate pure diffusion emergence.
2. Validate stochastic diversity.
3. Validate financial realism.
4. Then scale architecture or training.

The immediate run should be fresh pure reconstruction training. Old checkpoints
are no longer semantically valid because conditioning, normalization, session
filtering, target scale, and encoder behavior changed.

Use:

```powershell
.\.venv\Scripts\python.exe run_pure_recon.py
```

Do not start from structural losses. First answer whether coherent denoising
geometry appears naturally.

## Edited Files And Rationale

### Core Data And Training

- `dataset.py`
  - Added live session-pure start-index filtering.
  - Requires the whole context and 20-step future to remain in one `session_id`.
  - Requires every future target flag to be valid.
  - Computes `target_scale` from the same valid training windows used by the live trainer.
  - Why: the previous training dataset still crossed sessions and invalidated emergence claims.

- `config.py`
  - Changed `features_file` to `normalized_feature_tensor.npy`.
  - Added structural-loss weights for later staged phases: smoothness, autocorr, tail.
  - Why: normalized features are now the default training input; structural weights are available but disabled in pure reconstruction launchers.

- `scripts/data_creation/normalize_features.py`
  - Reworked robust scaling to fit only on the training period before 2023-01-01 UTC.
  - Writes a current-shape `data/normalized_feature_tensor.npy`.
  - Uses chunked writing to avoid large full-memory copies.
  - Why: feature normalization must not leak validation/test statistics and must match current data geometry.

- `trainer.py`
  - Allows `x0_pred` to carry gradients only when structural weights are active.
  - Keeps pure reconstruction mode cheap and clean when all structural weights are zero.
  - Why: structural losses need gradients later, but Stage A should remain pure diffusion reconstruction.

- `losses.py`
  - Added optional best-of-K path-level losses for later phases: volatility, trend, direction continuity, smoothness, autocorrelation, tail moment, diversity, manifold spread, latent sensitivity.
  - Pure reconstruction launchers explicitly zero these weights.
  - Why: later phases need controlled refinement tools, but emergence validation comes first.

- `run_pure_recon.py`
  - Explicitly disables every structural/financial/manifold loss.
  - Why: fresh Stage A must test denoising geometry only.

- `continue_pure_recon.py`
  - Explicitly disables every structural/financial/manifold loss.
  - Why: prevents accidental non-pure continuation.

- `run_staged_training.py`
  - Added gradual structural-loss schedule slots.
  - Stage A is pure reconstruction.
  - Later stages can add categories progressively.
  - Why: avoid over-regularized curve fitting.

### Model Conditioning

- `behavioral_encoder.py`
  - Added explicit regime summary projection into `B0`.
  - Summaries include trend, volatility, RSI, liquidity/session, recent-vs-early volatility, skew/tail proxies.
  - Why: conditioning should expose regime state without requiring the GRU to infer all regimes implicitly.

### Evaluation And Analysis

- `evaluate.py`
  - Unscales targets before comparing to generated raw-return paths.
  - Adds composite best-path criteria using direction, magnitude, volatility, and structure.
  - Why: previous evaluation compared scaled targets to unscaled generated paths.

- `scripts/analysis/analyze_samples.py`
  - Unscales validation targets before generated-vs-real statistics.
  - Why: analysis metrics need consistent units.

- `scripts/analysis/visualize.py`
  - Unscales validation targets before plotting against generated paths.
  - Why: visual overlays need consistent units.

- `scripts/analysis/emergence_dashboard.py`
  - New read-only checkpoint dashboard.
  - Produces generated overlays, latent diversity panels, volatility proxy comparison, return distribution comparison, DDIM trajectory snapshots, and autocorrelation decay.
  - Writes `emergence_summary.json`.
  - Why: emergence validation needs visual and quantitative diagnostics beyond loss.

### Repository Organization

- `scripts/data_creation/`
  - Moved data download, cleaning, feature building, normalization, and optional packed-window scripts here.

- `scripts/validation/`
  - Moved OHLCV/dataset/target/event validation scripts here.

- `scripts/analysis/`
  - Moved visualization, sample analysis, Numba metric processing, denoising diagnostics, and emergence dashboard here.

- `temp/scratch_scripts/`
  - Moved scratch probes and one-off test scripts here instead of deleting them.

- `requirements.txt`
  - Added `pandas` and `requests`.
  - Why: existing scripts import them and fresh environments need reproducible dependencies.

## Current Known Boundaries

- Core modules remain at repo root for now: `config.py`, `dataset.py`, `generator.py`,
  `behavioral_encoder.py`, `diffusion_unet.py`, `diffusion_sampler.py`, `losses.py`,
  `trainer.py`, `metrics_tracker.py`, `safety.py`, `evaluate.py`.
- This is intentional during emergence validation to avoid import churn.
- `old/` remains an archive, not part of the active workflow.
- `temp/scratch_scripts/` is not part of the active workflow.

## Latest Verified State

- Live dataset contamination check:
  - train context/future crossings: `0`
  - val context/future crossings: `0`
  - test context/future crossings: `0`
  - invalid 20-step target windows: `0`
- Live split sizes after filtering:
  - train: `3,351,462`
  - val: `333,072`
  - test: `238,710`
- Normalized feature tensor:
  - shape: `(4174320, 42)`
  - finite check passed
- Model smoke:
  - `debug_forward.py` passed
  - generated path shape: `[2, 4, 20]`
- Structural-loss backward sanity:
  - passed
- Compile check:
  - passed for core and launch files

## Emergence Validation Checklist

Before adding structural losses, inspect:

- DDIM denoising evolution under fixed seed and fixed context.
- Same-context multi-path diversity from varying `z`.
- Variance collapse or path convergence.
- High-noise timestep reconstruction behavior.
- Generated return moments versus real return moments.
- Volatility clustering proxy.
- Autocorrelation decay.
- Turning-point and sign persistence.

Only if pure diffusion shows coherent stochastic futures should Phase B begin.

## Stage A Emergence Readout - 2026-05-21

Checkpoint inspected:

```text
checkpoints/pure_recon/step_5000_final.pt
```

Visual read:

- Generated paths are not collapsed; same-context latent variation is visible.
- Path bundle is too wide relative to most real 20-step targets.
- DDIM denoising trajectory changes amplitude more than geometry, so denoising emergence exists but is still shallow.
- Generated paths are noisy and high-energy compared with real paths.

Metric read:

- Real std from emergence sample: about `0.000233`.
- Generated std: about `0.000568`.
- Generated variance ratio from sample analysis: about `5.74`.
- Real kurtosis: about `7.95` to `11.62` depending on sample set.
- Generated kurtosis: about `3.07` in sample analysis and near Gaussian in the 8-context dashboard.
- Generated volatility clustering is present but weaker than real: about `0.133` vs real `0.224`.

Interpretation:

- Stage A passed the basic denoising/stochasticity gate.
- Stage A did not pass financial calibration.
- The next step is Phase B1 only: tiny volatility consistency and direction continuity.
- Do not add autocorrelation, diversity, manifold spread, smoothness, or tail losses yet.

Execution plan:

1. Patch evaluation so success is calibrated by target variance, not raw `0.01` MSE.
2. Start Phase B1 from Stage A weights only, with fresh optimizer/scheduler.
3. Train B1 for a short controlled run.
4. Re-run emergence dashboard and calibrated evaluation.
5. Accept B1 only if variance ratio drops without path collapse.

New command:

```powershell
python.exe run_phase_b1.py
```

Acceptance targets for B1:

- Mean/median variance ratio moves toward `1-2.5`, not below `0.75`.
- Latent diversity remains visible in overlays.
- Tight cone coverage improves without all paths becoming smooth copies.
- Best-path scaled MSE improves materially from Stage A.
- DDIM trajectory still shows stochastic path formation rather than deterministic curve fitting.

Calibrated Stage A evaluation snapshot:

```text
samples=300
raw_generator_success=100.00%
raw_cone_coverage=100.00%
tight_cone_coverage=100.00%
mean_scaled_closest_distance=5.3869
calibrated_success_scaled_mse_lt_1=0.67%
mean_variance_ratio=13.975
median_variance_ratio=12.222
direction_coverage=100.00%
best_mse_direction_match=52.67%
```

Decision:

- Raw success/coverage is not meaningful yet because generated paths are too wide.
- B1 is approved as a short corrective refinement because the model is diverse but over-dispersed.
- B1 must reduce variance ratio without collapsing latent diversity.
