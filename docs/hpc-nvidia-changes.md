# Changes From Upstream

This document summarizes the current repository state compared with
`upstream/main` from `https://github.com/leigest519/HiddenDetect`.

Comparison base: `upstream/main` at `2fd161b` (`Add files via upload`).
Current branch: `hpc-nvidia`.

## Summary

The branch adds an NVIDIA HPC execution path, makes LLaVA and Qwen
evaluation configurable from the command line, adds dataset download and
verification tooling, and adds tests around the new runtime helpers.

It also changes LLaVA scoring. Upstream used a fixed trapezoid score over
selected layers. This branch added supervised Fisher/logistic-regression
scoring, and the current repository now evaluates those supervised modes with
out-of-fold scores to avoid training and testing on the same examples.

## Evaluation Changes

- `code/test.py` now accepts runtime options instead of hard-coded defaults:
  `--model-path`, `--output-path`, `--limit`, `--seed`, and `--device`.
- LLaVA tensor placement is device-aware. The code can run on CUDA when
  available and can fail clearly when CUDA is requested but unavailable.
- LLaVA scoring is selected with `--scoring-mode {trapz,fisher,logreg}`.
  - `trapz` preserves the upstream-style unsupervised layer AUC score.
  - `fisher` learns Fisher layer weights with stratified out-of-fold scoring.
  - `logreg` trains logistic regression with stratified out-of-fold scoring.
- `--n-folds` controls the fold count for supervised scoring. The effective
  fold count is capped by the smallest class count, and supervised scoring
  errors when a class has fewer than two examples.
- `code/eval_scoring.py` contains the reusable scoring implementation so the
  leakage-sensitive behavior can be unit-tested without importing LLaVA.
- `code/test_qwen.py` now accepts the same runtime options except for
  `--scoring-mode` and `--n-folds`; it keeps upstream-style trapezoid scoring.
- `code/eval_runtime.py` centralizes model path validation and result CSV
  writing. It refuses to write a header-only CSV if every dataset fails.

## HPC NVIDIA Support

- `scripts/hpc/setup_nvidia_env.sh` creates or reuses a Conda environment,
  installs a pinned local LLaVA checkout under `src/llava`, and installs this
  repository's requirements.
- `scripts/hpc/check_nvidia_cuda.py` verifies CUDA visibility, required Python
  imports, GPU properties, and a small CUDA tensor operation.
- `scripts/slurm/run_llava_eval.sbatch` runs LLaVA evaluation on an NVIDIA GPU
  allocation, activates the configured Conda environment, validates CUDA, and
  writes results under `results/` by default.
- `scripts/slurm/run_qwen_eval.sbatch` provides the same SLURM flow for Qwen.
- The SLURM scripts support environment overrides including `CONDA_ENV`,
  `CONDA_ENV_PREFIX`, `MODEL_PATH`, `OUTPUT_PATH`, and `LIMIT`.
- `.gitignore` now excludes generated logs, result CSVs, staged model
  directories, temporary dataset downloads, and large local image/data outputs.

## Dataset Tooling

- `scripts/download_datasets.py` downloads or stages the datasets consumed by
  `code/load_datasets.py`.
- `scripts/verify_datasets.py` checks required dataset files, columns,
  parquet readability, JSON metadata, image references, and common missing-file
  cases.
- The downloader is resumable and skips existing files unless overwrite is
  requested.
- `requirements.txt` now includes `datasets==3.2.0` for Hugging Face dataset
  loading and removes the direct editable LLaVA Git dependency. LLaVA is now
  installed by the HPC setup script from the pinned local checkout.

## Documentation

- `README.md` adds an HPC NVIDIA section with setup, CUDA verification, smoke
  runs, full runs, H200 submission examples, and model/output overrides.
- `docs/hpc-nvidia-runbook.md` gives the end-to-end workflow for environment
  setup, model staging, dataset preparation, SLURM submission, and
  troubleshooting.
- `docs/hpc-nvidia-plan.md` records the implementation plan used for the HPC
  work.
- This changelog replaces the earlier file-level summary with a current
  behavior-level comparison against upstream.

## Tests Added

- `tests/test_eval_runtime.py` covers model path validation, result writing,
  failed-dataset reporting, and header-only CSV prevention.
- `tests/test_eval_scoring.py` covers trapezoid scoring, out-of-fold logistic
  regression scoring, out-of-fold Fisher scoring, and invalid small-class
  supervised evaluation.
- `tests/test_download_datasets.py` covers dataset downloader helpers and
  selected download/verification flows.
- `tests/test_verify_datasets.py` covers dataset verifier helpers and error
  reporting.

## Public Interface Changes

New LLaVA CLI flags in `code/test.py`:

- `--model-path`
- `--output-path`
- `--limit`
- `--seed`
- `--device`
- `--scoring-mode {trapz,fisher,logreg}`
- `--n-folds`

New Qwen CLI flags in `code/test_qwen.py`:

- `--model-path`
- `--output-path`
- `--limit`
- `--seed`
- `--device`

New HPC environment variables:

- `CONDA_ENV`
- `CONDA_ENV_PREFIX`
- `MODEL_PATH`
- `OUTPUT_PATH`
- `LIMIT`
- Setup-only overrides: `PYTHON_VERSION`, `LLAVA_REPO`, `LLAVA_DIR`,
  `LLAVA_COMMIT`, and `RUN_CUDA_CHECK`

## Validation

Recent local validation:

- `conda run -n llava python -m py_compile code/eval_scoring.py code/test.py`
- `conda run -n llava python -m unittest discover`

The full test discovery run completed 18 tests successfully.
