# MMFPS_GEN_V2

Behavioral diffusion research pipeline for generating multiple plausible
financial future trajectories conditioned on historical market context.

The project is currently in **pure diffusion emergence validation**. The next
valid experiment is a fresh pure reconstruction run; old checkpoints should be
treated as incompatible with the current data/model semantics.

## Repository Layout

- `config.py`, `dataset.py`, `generator.py`, `trainer.py`
  - active core training and model surface.
- `behavioral_encoder.py`, `diffusion_unet.py`, `diffusion_sampler.py`
  - active model internals.
- `losses.py`, `metrics_tracker.py`, `safety.py`, `evaluate.py`
  - active training/evaluation support.
- `scripts/data_creation/`
  - raw download, cleaning, feature building, normalization, optional window packing.
- `scripts/validation/`
  - data quality and dataset validation scripts.
- `scripts/analysis/`
  - visualization, sample analysis, emergence dashboard, offline metric processing.
- `temp/scratch_scripts/`
  - quarantined scratch probes, not part of the main workflow.
- `old/`
  - archived previous implementation.

## Environment

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Use the repo root as the working directory for all commands below.

## Data Pipeline Commands

Download raw XAUUSD data with BI5 parser:

```powershell
.\.venv\Scripts\python.exe -m scripts.data_creation.download_xauusd
```

Alternative downloader using `aria2c`:

```powershell
.\.venv\Scripts\python.exe -m scripts.data_creation.download_xauusd_aria2
```

Alternative `dukascopy-node` batch downloader:

```powershell
.\.venv\Scripts\python.exe -m scripts.data_creation.download_xauusd_batch
```

Clean raw OHLCV:

```powershell
.\.venv\Scripts\python.exe -m scripts.data_creation.clean_ohlcv
```

Build raw feature and target tensors:

```powershell
.\.venv\Scripts\python.exe -m scripts.data_creation.build_features
```

Normalize features with train-period-only robust scaling:

```powershell
.\.venv\Scripts\python.exe -m scripts.data_creation.normalize_features
```

Optional legacy packed-window builder:

```powershell
.\.venv\Scripts\python.exe -m scripts.data_creation.build_diffusion_windows
```

The active trainer uses the live flat tensor dataset through `dataset.py`, not
the packed-window artifact.

## Validation Commands

Validate cleaned OHLCV:

```powershell
.\.venv\Scripts\python.exe -m scripts.validation.validate_ohlcv
```

Validate optional packed diffusion dataset:

```powershell
.\.venv\Scripts\python.exe -m scripts.validation.validate_diffusion_dataset
```

Inspect diffusion dataset:

```powershell
.\.venv\Scripts\python.exe -m scripts.validation.inspect_diffusion_dataset
```

Check event structure:

```powershell
.\.venv\Scripts\python.exe -m scripts.validation.check_events
```

Diagnose targets:

```powershell
.\.venv\Scripts\python.exe -m scripts.validation.diagnose_targets
```

Quick model forward smoke:

```powershell
.\.venv\Scripts\python.exe debug_forward.py
```

## Training Commands

Fresh pure reconstruction training, current required next step:

```powershell
.\.venv\Scripts\python.exe run_pure_recon.py
```

Continue pure reconstruction from the configured checkpoint:

```powershell
.\.venv\Scripts\python.exe continue_pure_recon.py
```

Staged training launcher:

```powershell
.\.venv\Scripts\python.exe run_staged_training.py --stage A --batch-size 512 --lr 5e-5 --output-dir checkpoints
```

Useful staged-training flags:

- `--stage A|B|C|D|all`
- `--batch-size <int>`
- `--lr <float>`
- `--output-dir <path>`
- `--resume-from <checkpoint.pt>`
- `--dry-run`

For now, prefer Stage A/pure reconstruction until emergence is confirmed.

## Evaluation And Emergence Commands

Evaluate a checkpoint:

```powershell
.\.venv\Scripts\python.exe evaluate.py --checkpoint checkpoints\pure_recon\step_5000_final.pt --num-paths 128 --max-samples 1000 --output checkpoints\pure_recon\eval.json
```

Evaluation flags:

- `--checkpoint <checkpoint.pt>`
- `--num-paths <int>`
- `--max-samples <int>`
- `--output <json path>`

Build the emergence dashboard:

```powershell
.\.venv\Scripts\python.exe -m scripts.analysis.emergence_dashboard --checkpoint checkpoints\pure_recon\step_5000_final.pt --output-dir checkpoints\pure_recon\emergence --split val --num-contexts 4 --num-paths 128 --seed 1234
```

Dashboard flags:

- `--checkpoint <checkpoint.pt>`
- `--output-dir <directory>`
- `--split train|val|test`
- `--num-contexts <int>`
- `--num-paths <int>`
- `--seed <int>`
- `--device cuda|cpu`

Analyze generated samples:

```powershell
.\.venv\Scripts\python.exe -m scripts.analysis.analyze_samples --checkpoint checkpoints\pure_recon\step_5000_final.pt --output checkpoints\pure_recon\sample_analysis.json --num-paths 128 --num-samples 100
```

Visualize generated paths:

```powershell
.\.venv\Scripts\python.exe -m scripts.analysis.visualize --checkpoint checkpoints\pure_recon\step_5000_final.pt --output-dir checkpoints\pure_recon\viz_manual --num-samples 4 --num-paths 128 --data-index 0
```

Compute offline structural metrics for generated chunks:

```powershell
.\.venv\Scripts\python.exe -m scripts.analysis.compute_metrics_numba --input-dir <generated_npz_dir> --output-dir <metrics_npz_dir>
```

## Current Phase Rules

- Do not add more losses before pure diffusion emergence is inspected.
- Do not resume old checkpoints for final conclusions.
- Watch DDIM trajectory, latent diversity, variance collapse, and high-noise timestep behavior.
- Only add structural losses gradually after pure diffusion produces believable stochastic futures.

