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

## Phase B1 Readout - 2026-05-21

Checkpoint inspected:

```text
checkpoints/phase_b1/step_1500_final.pt
```

What changed:

- Added `run_phase_b1.py` as a controlled refinement launcher.
- Continued from `checkpoints/pure_recon/step_5000_final.pt` with model weights only.
- Used a fresh optimizer/scheduler to avoid stale Stage A optimizer state.
- Enabled only tiny structural pressure:
  - volatility consistency: `0.03`
  - turning/direction continuity: `0.01`
- Kept all heavier financial realism, autocorrelation, diversity, and tail losses disabled.

Training result:

```text
steps=1500
final_loss=0.271
final_reconstruction=0.269
final_nmse=0.2695
final_grad_norm=0.55
final_coverage=90.62%
```

Calibrated evaluation comparison:

```text
Stage A checkpoint: checkpoints/pure_recon/step_5000_final.pt
B1 checkpoint:      checkpoints/phase_b1/step_1500_final.pt

mean_scaled_closest_distance: 5.3869 -> 4.9615
calibrated_success_mse_lt_1:  0.67%  -> 0.33%
mean_variance_ratio:          13.975 -> 13.355
median_variance_ratio:        12.222 -> 11.611
direction_coverage:           100%   -> 100%
best_mse_direction_match:     52.67% -> 54.67%
```

Sample analysis comparison:

```text
variance_ratio_gen_real: 5.743 -> 5.484
scaled_best_mse:        65.92 -> 63.84
gen_vol_cluster_lag1:   0.133 -> 0.138
real_vol_cluster_lag1:  0.224
gen_kurtosis:           3.07 -> 3.11
real_kurtosis:          7.95
KS_stat:                0.2755 -> 0.2702
```

Interpretation:

- B1 moved in the correct direction, but only weakly.
- Generated futures remain over-dispersed.
- Generated returns are still too Gaussian relative to real heavy-tailed targets.
- Volatility clustering improved slightly but remains below real data.
- Directional coverage remains high, but raw coverage is still inflated by overly wide path bundles.
- The model has not collapsed; stochasticity is still alive.

Decision:

- Do not add B2 losses yet.
- Do not scale architecture yet.
- Next step is a visual B1 check, then a slightly longer B1 continuation only if overlays confirm diversity remains healthy.
- If B1 continuation still barely reduces variance, investigate sampling/calibration before adding more losses.

Next commands:

```powershell
python.exe -m scripts.analysis.visualize --checkpoint checkpoints\phase_b1\step_1500_final.pt --output-dir checkpoints\phase_b1\viz_step_1500 --num-samples 8 --num-paths 128 --data-index 0
python.exe -m scripts.analysis.emergence_dashboard --checkpoint checkpoints\phase_b1\step_1500_final.pt --output-dir checkpoints\phase_b1\emergence --split val --num-contexts 8 --num-paths 128 --seed 1234
python.exe evaluate.py --checkpoint checkpoints\phase_b1\step_1500_final.pt --num-paths 128 --max-samples 1000 --output checkpoints\phase_b1\eval_step_1500_calibrated_1000.json
```

## Phase B1 Extended Validation - 2026-05-21

Executed:

```powershell
python.exe -m scripts.analysis.visualize --checkpoint checkpoints\phase_b1\step_1500_final.pt --output-dir checkpoints\phase_b1\viz_step_1500 --num-samples 8 --num-paths 128 --data-index 0
python.exe evaluate.py --checkpoint checkpoints\phase_b1\step_1500_final.pt --num-paths 128 --max-samples 1000 --output checkpoints\phase_b1\eval_step_1500_calibrated_1000.json
```

Artifacts:

```text
checkpoints/phase_b1/viz_step_1500/
checkpoints/phase_b1/viz_step_1500/denoising_progression.png
checkpoints/phase_b1/eval_step_1500_calibrated_1000.json
```

1000-sample calibrated evaluation:

```text
raw_generator_success=100.00%
raw_cone_coverage=100.00%
tight_cone_coverage=96.80%
mean_scaled_closest_distance=2.9386
calibrated_success_scaled_mse_lt_1=5.70%
mean_variance_ratio=7.254
median_variance_ratio=4.830
direction_coverage=100.00%
best_mse_direction_match=60.00%
```

Visual read:

- Generated paths still contain the real path within the 128-path bundle.
- Latent diversity remains visible; B1 did not collapse stochasticity.
- The generated family remains too wide relative to most real futures.
- Return distribution is still too smooth/Gaussian compared with real heavy-tailed returns.
- Volatility clustering remains weaker than real.
- DDIM denoising still looks like amplitude refinement around a mostly fixed shape, not a rich progressive geometry change.

Decision:

- B1 is promising but incomplete.
- Do not move to B2 yet.
- Do not add autocorrelation, smoothness, tail, diversity, or manifold-spread losses yet.
- Continue B1 only, or investigate sampler calibration/target scaling if B1 continuation plateaus.

Recommended immediate next step:

```powershell
python.exe run_phase_b1.py --resume checkpoints\phase_b1\step_1500_final.pt
```

Implementation note:

- `run_phase_b1.py` currently starts from the Stage A checkpoint by default.
- Add CLI override flags before using the command above, or create a `run_phase_b1_continue.py` launcher that resumes from `checkpoints/phase_b1/step_1500_final.pt` with the same tiny B1 loss weights.

## Phase B1 Continuation - 2026-05-21

Code change:

- Updated `run_phase_b1.py` with CLI overrides:
  - `--resume`
  - `--output-dir`
  - `--steps`
  - `--batch-size`
  - `--lr`
  - `--warmup-steps`
  - `--training-paths-per-sample`
  - `--vol-weight`
  - `--turning-weight`
  - `--full-resume`
- Default behavior remains Stage A -> B1 weights-only training.
- Continuation probes can now be run without editing source constants.

Executed:

```powershell
python.exe run_phase_b1.py --resume checkpoints\phase_b1\step_1500_final.pt --output-dir checkpoints\phase_b1_continue --steps 1500
python.exe evaluate.py --checkpoint checkpoints\phase_b1_continue\step_1500_final.pt --num-paths 128 --max-samples 1000 --output checkpoints\phase_b1_continue\eval_step_1500_calibrated_1000.json
python.exe -m scripts.analysis.analyze_samples --checkpoint checkpoints\phase_b1_continue\step_1500_final.pt --output checkpoints\phase_b1_continue\sample_analysis_step_1500.json --num-paths 128 --num-samples 250
python.exe -m scripts.analysis.visualize --checkpoint checkpoints\phase_b1_continue\step_1500_final.pt --output-dir checkpoints\phase_b1_continue\viz_step_1500 --num-samples 8 --num-paths 128 --data-index 0
python.exe -m scripts.analysis.emergence_dashboard --checkpoint checkpoints\phase_b1_continue\step_1500_final.pt --output-dir checkpoints\phase_b1_continue\emergence --split val --num-contexts 8 --num-paths 128 --seed 1234
```

Continuation checkpoint:

```text
checkpoints/phase_b1_continue/step_1500_final.pt
```

Training result:

```text
steps=1500
final_loss=0.313
final_reconstruction=0.312
final_nmse=0.3122
final_grad_norm=0.40
final_coverage=96.88%
```

1000-sample evaluation comparison:

```text
B1:          checkpoints/phase_b1/step_1500_final.pt
B1 continue: checkpoints/phase_b1_continue/step_1500_final.pt

mean_scaled_closest_distance: 2.9386 -> 2.7176
calibrated_success_mse_lt_1:  5.70%  -> 6.90%
mean_variance_ratio:          7.254  -> 6.876
median_variance_ratio:        4.830  -> 4.579
tight_cone_coverage:          96.80% -> 96.60%
best_mse_direction_match:     60.00% -> 61.80%
```

Sample analysis comparison:

```text
variance_ratio_gen_real: 5.484 -> 5.172
scaled_best_mse:        63.84 -> 56.75
gen_vol_cluster_lag1:   0.138 -> 0.140
real_vol_cluster_lag1:  0.224
gen_kurtosis:           3.11 -> 3.14
real_kurtosis:          7.95
KS_stat:                0.2702 -> 0.2641
```

Dashboard moment comparison:

```text
generated_std:          0.000554 -> 0.000538
real_std:               0.000233
generated_kurtosis_excess: 0.098 -> 0.146
real_kurtosis_excess:      11.621
generated_tail_p95_abs: 0.001100 -> 0.001072
real_tail_p95_abs:      0.000584
```

Interpretation:

- Continuation improved calibration again, but the slope is shallow.
- Generated paths remain too wide.
- Heavy-tail behavior is still missing.
- Volatility clustering remains weaker than real.
- Direction matching improved.
- Stochastic diversity is still present; no path collapse observed.

Decision:

- Best current checkpoint for evaluation is `checkpoints/phase_b1_continue/step_1500_final.pt`.
- Do not start B2 yet.
- Do not add tail/autocorrelation/diversity/manifold losses yet.
- Next best move is a sampling/calibration audit before more training:
  - compare DDIM steps and eta/noise settings,
  - test generated variance versus sampler step count,
  - verify no double target-scale amplification exists in generator/evaluator paths,
  - inspect whether B1 is reducing variance through model learning or only through sample smoothing.

Next recommended command block:

```powershell
python.exe -m scripts.analysis.denoising_diagnostics --checkpoint checkpoints\phase_b1_continue\step_1500_final.pt --output-dir checkpoints\phase_b1_continue\denoising_diag
python.exe -m scripts.analysis.visualize --checkpoint checkpoints\phase_b1_continue\step_1500_final.pt --output-dir checkpoints\phase_b1_continue\viz_step_1500_check --num-samples 16 --num-paths 128 --data-index 0
python.exe evaluate.py --checkpoint checkpoints\phase_b1_continue\step_1500_final.pt --num-paths 128 --max-samples 5000 --output checkpoints\phase_b1_continue\eval_step_1500_calibrated_5000.json
```

## Phase B1 5000-Sample Audit - 2026-05-21

User executed:

```powershell
python.exe -m scripts.analysis.denoising_diagnostics --checkpoint checkpoints\phase_b1_continue\step_1500_final.pt --output-dir checkpoints\phase_b1_continue\denoising_diag
python.exe -m scripts.analysis.visualize --checkpoint checkpoints\phase_b1_continue\step_1500_final.pt --output-dir checkpoints\phase_b1_continue\viz_step_1500_check --num-samples 16 --num-paths 128 --data-index 0
python.exe evaluate.py --checkpoint checkpoints\phase_b1_continue\step_1500_final.pt --num-paths 128 --max-samples 5000 --output checkpoints\phase_b1_continue\eval_step_1500_calibrated_5000.json
```

Important tooling observation:

- `scripts.analysis.denoising_diagnostics` currently contains helper functions only.
- It does not have an active CLI entrypoint, so the command produced no diagnostic artifacts.
- This should be fixed before relying on denoising diagnostics as a real gate.

5000-sample calibrated evaluation:

```text
raw_generator_success=100.00%
raw_cone_coverage=100.00%
tight_cone_coverage=98.12%
mean_scaled_closest_distance=2250.0458
calibrated_success_scaled_mse_lt_1=6.14%
mean_variance_ratio=7108.315
median_variance_ratio=8.295
direction_coverage=97.44%
best_mse_direction_match=58.86%
```

Interpretation:

- The 5000-sample run changes the story.
- Median variance ratio is only `8.295`, but mean variance ratio is `7108.315`.
- Mean scaled closest distance also exploded to `2250.0458`.
- This means evaluation is dominated by rare catastrophic outliers, not by the typical case.
- Tight cone coverage remains very high because the generated envelope is still too wide.
- Calibrated success remains low at `6.14%`.
- Direction coverage dropped from `100%` on small samples to `97.44%`, so some contexts do not contain the correct direction even among 128 futures.

Visual read from `viz_step_1500_check`:

- Generated paths remain diverse; there is no obvious global collapse.
- The generated family is still wider than real targets.
- Closest paths often match envelope inclusion but miss local structure.
- DDIM progression remains mostly shape-preserving amplitude refinement, not a strong progressive denoising geometry.

Decision:

- Do not continue training yet.
- Do not start B2.
- Do not add heavier structural losses.
- The next blocker is outlier diagnosis and sampler/scale audit.

Code change:

- Updated `evaluate.py` to emit:
  - scaled closest-distance percentiles,
  - variance-ratio percentiles,
  - outlier rates above `10x`, `100x`, and `1000x`,
  - top worst samples by scaled error,
  - top worst samples by variance ratio.
- This is needed because mean metrics are unstable and hide the real failure mode.

Immediate next plan:

1. Re-run evaluation with the enhanced evaluator and inspect the worst sample indices.
2. Visualize the top 5 worst-by-scaled-error and worst-by-variance contexts directly.
3. Check whether outliers are caused by:
   - near-zero target variance,
   - generated path explosions,
   - target-scale mismatch,
   - sampler stochasticity,
   - invalid/abnormal validation windows.
4. Only after this decide between:
   - sampler calibration,
   - target variance floor change,
   - dataset outlier filtering,
   - continued B1 training.

Next commands:

```powershell
python.exe evaluate.py --checkpoint checkpoints\phase_b1_continue\step_1500_final.pt --num-paths 128 --max-samples 5000 --output checkpoints\phase_b1_continue\eval_step_1500_calibrated_5000_outliers.json
python.exe -m scripts.analysis.visualize --checkpoint checkpoints\phase_b1_continue\step_1500_final.pt --output-dir checkpoints\phase_b1_continue\viz_outlier_probe --num-samples 16 --num-paths 128 --data-index 0
```

## Flat-Target Contamination Fix - 2026-05-21

Trigger:

The enhanced 5000-sample evaluator showed:

```text
mean_scaled_closest_distance=2259.9899
calibrated_success_scaled_mse_lt_1=5.66%
mean_variance_ratio=7090.140
median_variance_ratio=8.324
scaled_distance_p50/p95/p99=3.068 / 17.471 / 99934.048
variance_ratio_p50/p95/p99=8.324 / 50.418 / 290239.321
```

Outlier diagnosis:

- Top outliers all had `target_std=0.0`.
- Top outliers all had `target_abs_max=0.0`.
- Their `target_var` was only the evaluator epsilon floor, about `1e-12`.
- Generated paths were not numerically exploding; generated absolute max was about `0.0018-0.0025`.
- The metric explosion came from dividing normal generated variance by zero-variance future targets.

Dataset contamination counts before the fix:

```text
train zero-std target windows: 1,012,949 / 3,351,462 = 30.22%
val zero-std target windows:     113,365 /   333,072 = 34.04%
test zero-std target windows:     70,588 /   238,710 = 29.57%
```

Root cause:

- `target_valid` marked flat 20-step future paths as valid.
- Session and temporal leakage checks passed, but target activity was not enforced.
- The model trained on a large number of stale/flat futures.
- Evaluation on variance-normalized metrics was therefore unstable and misleading.

Code changes:

- `config.py`
  - Added `min_target_std: float = 1e-8`.
- `dataset.py`
  - `_compute_valid_starts` now receives `targets`.
  - Computes each candidate future path standard deviation using cumulative sums.
  - Rejects windows where future target std is below `min_target_std`.
  - `build_splits` now computes target scale from active training futures only.

Post-fix dataset validation:

```text
train windows: 2,338,513
val windows:     219,707
test windows:    168,122
target_scale from active train windows: 2690.8517

train zero/near-zero target windows: 0
val zero/near-zero target windows:   0
test zero/near-zero target windows:  0
```

Post-fix checkpoint probe:

Executed against the old B1-continuation checkpoint on the cleaned validation set:

```powershell
python.exe evaluate.py --checkpoint checkpoints\phase_b1_continue\step_1500_final.pt --num-paths 128 --max-samples 1000 --output checkpoints\phase_b1_continue\eval_step_1500_cleaned_val_1000.json
```

Result:

```text
mean_scaled_closest_distance=2.7283
calibrated_success_scaled_mse_lt_1=6.20%
mean_variance_ratio=6.881
median_variance_ratio=4.602
scaled_distance_p50/p95/p99=1.966 / 6.697 / 12.507
variance_ratio_p50/p95/p99=4.602 / 17.408 / 38.838
direction_coverage=100.00%
best_mse_direction_match=62.50%
```

Interpretation:

- The catastrophic p99 explosion was caused by flat-target contamination.
- The model is still over-dispersed, but the cleaned metric distribution is interpretable.
- Existing checkpoints are not final because they were trained with roughly 30% flat future windows.
- The next real phase must restart Stage A on the cleaned dataset.

Decision:

- Stop continuing B1 from the contaminated checkpoint.
- Do not start B2.
- Do not add heavier losses.
- Run fresh pure reconstruction training on the cleaned active-target dataset.
- Treat prior B1/B1-continuation results as diagnostic only.

Next commands:

```powershell
python.exe debug_forward.py
python.exe run_pure_recon.py --output-dir checkpoints\pure_recon_cleaned --steps 5000
python.exe evaluate.py --checkpoint checkpoints\pure_recon_cleaned\step_5000_final.pt --num-paths 128 --max-samples 5000 --output checkpoints\pure_recon_cleaned\eval_cleaned_step_5000.json
python.exe -m scripts.analysis.emergence_dashboard --checkpoint checkpoints\pure_recon_cleaned\step_5000_final.pt --output-dir checkpoints\pure_recon_cleaned\emergence --split val --num-contexts 8 --num-paths 128 --seed 1234
python.exe -m scripts.analysis.visualize --checkpoint checkpoints\pure_recon_cleaned\step_5000_final.pt --output-dir checkpoints\pure_recon_cleaned\viz_step_5000 --num-samples 16 --num-paths 128 --data-index 0
```

Acceptance gates for the fresh cleaned Stage A:

- No zero-target windows in train/val/test.
- No p99 scaled-distance explosion.
- Median variance ratio should be meaningfully lower than the contaminated checkpoint or at least stable without p99 blow-up.
- Calibrated success should improve without losing latent diversity.
- DDIM progression should still show stochastic path formation.

## 2026-05-22 - Cleaned Stage A Result and Controlled B1 Branch

Fresh cleaned pure reconstruction training completed:

```powershell
python.exe run_pure_recon.py --output-dir checkpoints\pure_recon_cleaned --steps 5000
```

Final checkpoint:

```text
checkpoints\pure_recon_cleaned\step_5000_final.pt
```

Training readout:

```text
train windows: 2,338,513
val windows:     219,707
test windows:    168,122
model params: 11.3M
final rec/nmse: 0.395 / 0.3954
final x0s: 1.3384
final grad norm: 0.58
final coverage: 93.75%
```

Full 5000-sample evaluation:

```powershell
python.exe evaluate.py --checkpoint checkpoints\pure_recon_cleaned\step_5000_final.pt --num-paths 128 --max-samples 5000 --output checkpoints\pure_recon_cleaned\eval_cleaned_step_5000.json
```

Result:

```text
generator_success=100.00%
cone_coverage=100.00%
tight_cone_coverage=99.40%
mean_scaled_closest_distance=7.6683
calibrated_success_scaled_mse_lt_1=2.36%
variance_ratio_mean/median=20.651 / 12.564
scaled_distance_p50/p95/p99=4.812 / 23.155 / 40.345
variance_ratio_p50/p95/p99=12.564 / 65.183 / 106.118
direction_coverage=99.90%
best_mse_direction_match=59.84%
```

Interpretation:

- The cleaned-data restart removed the earlier flat-window p99 catastrophe.
- The final 5000-step pure reconstruction checkpoint is not the best generator checkpoint.
- It is stable and diverse, but too over-dispersed for the "128 futures should contain a close future" objective.
- Final training loss alone is not a reliable checkpoint selector for this diffusion task.

Checkpoint screen on cleaned Stage A:

```text
checkpoint    mean scaled dist    success <1x    var ratio med    direction match
step_1000     2.2134              8.00%          4.616            63.67%
step_2000     2.4074              7.33%          4.546            59.33%
step_3000     2.4046              9.33%          4.694            58.67%
step_4000     2.4753              8.00%          4.613            56.00%
step_5000     7.6683              2.36%          12.564           59.84%
```

Decision:

- Use `checkpoints\pure_recon_cleaned\step_1000.pt` as the best current Stage A base.
- Do not use the final 5000-step checkpoint as the base for refinement.
- Begin Phase B1 from step 1000 using tiny volatility/turning regularization only.

Step-1000 emergence check:

```powershell
python.exe -m scripts.analysis.emergence_dashboard --checkpoint checkpoints\pure_recon_cleaned\step_1000.pt --output-dir checkpoints\pure_recon_cleaned\emergence_step_1000 --split val --num-contexts 8 --num-paths 128 --seed 1234
python.exe -m scripts.analysis.visualize --checkpoint checkpoints\pure_recon_cleaned\step_1000.pt --output-dir checkpoints\pure_recon_cleaned\viz_step_1000 --num-samples 16 --num-paths 128 --data-index 0
```

Step-1000 realism readout:

```text
generated std: 0.000343
real std:      0.000245
generated kurtosis: 0.497 dashboard excess / 3.47 sample kurtosis
real kurtosis:      3.909 dashboard excess / 7.95 sample kurtosis
generated autocorr lag1: -0.0688
real autocorr lag1:       0.0180 dashboard sample / -0.0443 broader sample
```

Interpretation:

- Step 1000 is the strongest pure checkpoint so far for closest-path containment.
- It remains too Gaussian and under-expresses volatility clustering/heavy tails.
- It is good enough to justify a controlled B1 refinement, but not good enough as a finished generator.

Executed cleaned Phase B1:

```powershell
python.exe run_phase_b1.py --resume checkpoints\pure_recon_cleaned\step_1000.pt --output-dir checkpoints\phase_b1_cleaned_from_1000 --steps 1500
```

Final checkpoint:

```text
checkpoints\phase_b1_cleaned_from_1000\step_1500_final.pt
```

B1 evaluation:

```powershell
python.exe evaluate.py --checkpoint checkpoints\phase_b1_cleaned_from_1000\step_1500_final.pt --num-paths 128 --max-samples 1000 --output checkpoints\phase_b1_cleaned_from_1000\eval_step_1500_1000.json
python.exe -m scripts.analysis.analyze_samples --checkpoint checkpoints\phase_b1_cleaned_from_1000\step_1500_final.pt --output checkpoints\phase_b1_cleaned_from_1000\sample_analysis_step_1500.json --num-paths 128 --num-samples 250
```

B1 result:

```text
generator_success=100.00%
cone_coverage=100.00%
tight_cone_coverage=86.90%
mean_scaled_closest_distance=1.3813
calibrated_success_scaled_mse_lt_1=35.30%
variance_ratio_mean/median=3.033 / 2.093
scaled_distance_p50/p95/p99=1.126 / 2.677 / 5.434
variance_ratio_p50/p95/p99=2.093 / 7.382 / 16.894
direction_coverage=100.00%
best_mse_direction_match=62.60%
```

B1 sample realism:

```text
generated std: 0.0004
real std:      0.0002
KS stat:       0.1951
variance ratio: 2.224
generated lag1 autocorr: -0.0668
real lag1 autocorr:      -0.0443
generated 2nd diff: 0.000688
real 2nd diff:      0.000368
generated vol cluster lag1: 0.1484
real vol cluster lag1:      0.2237
generated kurtosis: 3.49
real kurtosis:      7.95
scaled best MSE:    25.9195
```

Interpretation:

- B1 materially improved the containment/calibration objective.
- Mean scaled closest distance improved from the screened Stage A range near `2.2-2.5` to `1.3813`.
- Calibrated success below 1x improved from roughly `8-9%` to `35.30%`.
- Median variance ratio improved from roughly `4.6x` to `2.093x`.
- The model is still too Gaussian, still under-clusters volatility, and still produces paths that are rougher than real targets.
- Phase B1 helped the core generator objective but did not solve financial realism.

Current readiness assessment:

- Data/pipeline readiness for research: high.
- Pure diffusion emergence: validated enough to continue.
- 128-path containment: partially validated, improved by B1, not finished.
- Financial realism: not yet satisfactory.
- Architecture scaling: still premature until B1 is validated on larger eval and visuals.

Next required actions:

1. Run full 5000-sample evaluation on `phase_b1_cleaned_from_1000`.
2. Generate B1 dashboard and denoising visuals.
3. Compare pure step 1000 vs B1 step 1500 visually and statistically.
4. If full B1 eval holds, run a small B1 weight sweep before adding B2.
5. Do not add autocorrelation, tail, diversity, or architecture changes until B1 is confirmed stable.

Commands:

```powershell
python.exe evaluate.py --checkpoint checkpoints\phase_b1_cleaned_from_1000\step_1500_final.pt --num-paths 128 --max-samples 5000 --output checkpoints\phase_b1_cleaned_from_1000\eval_step_1500_5000.json
python.exe -m scripts.analysis.emergence_dashboard --checkpoint checkpoints\phase_b1_cleaned_from_1000\step_1500_final.pt --output-dir checkpoints\phase_b1_cleaned_from_1000\emergence --split val --num-contexts 8 --num-paths 128 --seed 1234
python.exe -m scripts.analysis.visualize --checkpoint checkpoints\phase_b1_cleaned_from_1000\step_1500_final.pt --output-dir checkpoints\phase_b1_cleaned_from_1000\viz_step_1500 --num-samples 16 --num-paths 128 --data-index 0
python.exe -m scripts.analysis.denoising_diagnostics --checkpoint checkpoints\phase_b1_cleaned_from_1000\step_1500_final.pt --output-dir checkpoints\phase_b1_cleaned_from_1000\denoising_diag
```

## 2026-05-29 - Live Emergence Dashboard

Goal:

- Add observability for diffusion emergence without changing the core generator architecture.
- Track 128 independent future paths for the same validation context across checkpoints.
- Preserve stable path identity using fixed latent vectors and fixed initial DDIM noise.
- Compare generated futures against the real historical future and expose denoising progression.

Files edited:

- `scripts/dashboard/__init__.py`
  - Added dashboard package marker.
- `scripts/dashboard/emergence_snapshots.py`
  - Added snapshot emitter, fixed validation scenario suite, fixed-latent/fixed-noise 128-path sampling, per-path metrics, denoising state capture, and self-contained local HTML dashboard generation.
- `trainer.py`
  - Added opt-in emergence snapshot callback.
  - Normal training remains unchanged unless `emergence_snapshot_every > 0`.
  - Snapshots are emitted after checkpoint saves and final save.
- `run_pure_recon.py`
  - Added CLI flags for emergence snapshots.
- `run_phase_b1.py`
  - Added the same CLI flags for B1 runs.
- `README.md`
  - Added dashboard commands and artifact contract.

Snapshot contract:

```text
scenario_suite.json
step_XXXXXX/snapshot.npz
  context:            (S, 128)
  real_future:        (S, 20)
  generated_futures:  (S, 128, 20)
  latent_z:           (S, 128, 128)
  denoising_states:   (S, snapshots, selected_paths, 20)
step_XXXXXX/metrics.json
dashboard.html
```

Dashboard features:

- Main context/real/best/ensemble comparison.
- 128 independent path tiles with stable path index.
- Per-path detail view and metrics.
- Best-of-K path highlighting.
- Denoising progression for paths `0-3`.
- Checkpoint-to-checkpoint training evolution cards.
- Optional local auto-refresh with `dashboard.html?live=1`.

Smoke checks executed:

```powershell
.\.venv\Scripts\python.exe -m py_compile trainer.py run_pure_recon.py run_phase_b1.py scripts\dashboard\emergence_snapshots.py
.\.venv\Scripts\python.exe -m scripts.dashboard.emergence_snapshots emit --checkpoint checkpoints\phase_b1_cleaned_from_1000\step_1500_final.pt --output-root visual_outputs\emergence_smoke --split val --num-scenarios 2 --num-paths 128 --seed 1234
.\.venv\Scripts\python.exe run_pure_recon.py --output-dir checkpoints\dashboard_hook_smoke --steps 1 --batch-size 2 --emergence-snapshot-every 1 --emergence-snapshot-dir visual_outputs\hook_smoke --emergence-num-scenarios 1 --emergence-num-paths 8 --emergence-seed 4321
```

Smoke output:

```text
visual_outputs\emergence_smoke\step_001500
visual_outputs\emergence_smoke\dashboard.html
visual_outputs\hook_smoke\step_000001
visual_outputs\hook_smoke\dashboard.html
```

Verified smoke snapshot shapes:

```text
context:           (2, 128)
real_future:       (2, 20)
generated_futures: (2, 128, 20)
latent_z:          (2, 128, 128)
denoising_states:  (2, 7, 4, 20)
```

Primary commands:

```powershell
.\.venv\Scripts\python.exe -m scripts.dashboard.emergence_snapshots emit --checkpoint checkpoints\phase_b1_cleaned_from_1000\step_1500_final.pt --output-root visual_outputs\emergence_live --split val --num-scenarios 8 --num-paths 128 --seed 1234
.\.venv\Scripts\python.exe run_pure_recon.py --output-dir checkpoints\pure_recon_live --steps 5000 --emergence-snapshot-every 1000 --emergence-snapshot-dir visual_outputs\pure_recon_live --emergence-num-scenarios 8 --emergence-num-paths 128 --emergence-seed 1234
.\.venv\Scripts\python.exe run_phase_b1.py --resume checkpoints\pure_recon_cleaned\step_1000.pt --output-dir checkpoints\phase_b1_live --steps 1500 --emergence-snapshot-every 500 --emergence-snapshot-dir visual_outputs\phase_b1_live --emergence-num-scenarios 8 --emergence-num-paths 128 --emergence-seed 1234
```

Important constraint preserved:

- No tokenizer work.
- No Kalman/HMM layer.
- No encoder redesign.
- No diffusion objective change.
- No new financial losses.
- Dashboard is checkpoint-driven observability only.
