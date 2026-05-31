# HPC NVIDIA Change Log

## Branch + scope

- Branch: `hpc-nvidia`
- Scope: adds an NVIDIA/HPC execution path while preserving existing default evaluation behavior when new options are not used.

## New files added

- `.gitignore`
  - Adds ignores for runtime/model/log/result artifacts and large staged data directories.
- `scripts/hpc/check_nvidia_cuda.py`
  - Verifies CUDA visibility, required imports, and a small CUDA tensor matmul check.
- `scripts/hpc/setup_nvidia_env.sh`
  - Creates/activates the `llava` Conda env, installs a local pinned LLaVA checkout, then installs HiddenDetect requirements for NVIDIA HPC runs.
- `scripts/slurm/run_llava_eval.sbatch`
  - SLURM entrypoint for LLaVA evaluation on NVIDIA GPUs.
- `scripts/slurm/run_qwen_eval.sbatch`
  - SLURM entrypoint for Qwen evaluation on NVIDIA GPUs.

## Existing files modified

- `code/test.py`
  - Added CLI args: `--model-path`, `--output-path`, `--limit`, `--seed`, `--device`.
  - Replaced CUDA-only assumptions with device-aware tensor movement.
  - Added output directory creation before writing CSV.
- `code/test_qwen.py`
  - Added CLI args: `--model-path`, `--output-path`, `--limit`, `--seed`, `--device`.
  - Replaced CUDA-only assumptions with device-aware model/input placement.
  - Added output directory creation before writing CSV.
- `README.md`
  - Added NVIDIA HPC usage section with smoke/full runs, H200 constraint example, and scheduler/model/output override examples.

## Public interface changes

- New CLI flags on both evaluation entrypoints (`code/test.py`, `code/test_qwen.py`):
  - `--model-path`
  - `--output-path`
  - `--limit`
  - `--seed`
  - `--device`
- New environment variables in batch flow:
  - `CONDA_ENV`
  - `MODEL_PATH`
  - `OUTPUT_PATH`
  - `LIMIT`
  - Optional setup-script overrides: `PYTHON_VERSION`, `LLAVA_REPO`, `LLAVA_DIR`, `LLAVA_COMMIT`, `RUN_CUDA_CHECK`

## Validation already performed

- Python syntax validation:
  - `python -m py_compile code/test.py code/test_qwen.py scripts/hpc/check_nvidia_cuda.py`
- Shell/SLURM script syntax validation:
  - `bash -n scripts/hpc/setup_nvidia_env.sh scripts/slurm/run_llava_eval.sbatch scripts/slurm/run_qwen_eval.sbatch`
