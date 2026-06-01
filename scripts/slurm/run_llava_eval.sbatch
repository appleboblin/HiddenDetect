#!/bin/bash
#SBATCH -J hiddendetect-llava
#SBATCH -p dgxh
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --mail-type=FAIL,END

set -euo pipefail

SUBMIT_CONDA_PREFIX="${CONDA_PREFIX:-}"
CONDA_ENV="${CONDA_ENV:-llava}"
MODEL_PATH="${MODEL_PATH:-model/llava-v1.6-vicuna-7b}"
OUTPUT_PATH="${OUTPUT_PATH:-results/llava-result.csv}"
LIMIT="${LIMIT:-}"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs results

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Host: $(hostname)"
echo "Repo: $(pwd)"
echo "Start: $(date)"

module purge
module load conda/25.3
module load cuda/12.1

source "$(conda info --base)/etc/profile.d/conda.sh"

if [[ -n "${CONDA_ENV_PREFIX:-}" ]]; then
  conda activate "${CONDA_ENV_PREFIX}"
elif [[ -n "${SUBMIT_CONDA_PREFIX}" && -d "${SUBMIT_CONDA_PREFIX}" ]]; then
  conda activate "${SUBMIT_CONDA_PREFIX}"
else
  conda activate "${CONDA_ENV}"
fi

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "ERROR: Conda activation did not set CONDA_PREFIX." >&2
  echo "Set CONDA_ENV_PREFIX=/full/path/to/the/env when submitting this job." >&2
  exit 1
fi

PYTHON_BIN="${CONDA_PREFIX}/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: Conda activation did not expose executable Python at: ${PYTHON_BIN}" >&2
  echo "Set CONDA_ENV_PREFIX=/full/path/to/the/env when submitting this job." >&2
  exit 1
fi

echo "Conda env: ${CONDA_DEFAULT_ENV:-unset}"
echo "Conda prefix: ${CONDA_PREFIX}"
echo "PATH Python: $(command -v python || true)"
echo "Env Python: ${PYTHON_BIN}"

"${PYTHON_BIN}" - <<'PY'
import importlib.util
import os
import site
import sys

print(f"sys.executable: {sys.executable}")
print(f"sys.prefix: {sys.prefix}")
print(f"CONDA_PREFIX: {os.environ.get('CONDA_PREFIX', 'unset')}")
print("site-packages:")
for path in site.getsitepackages():
    print(f"  {path}")

if importlib.util.find_spec("torch") is None:
    raise SystemExit(
        "Activated Python environment does not contain torch. "
        "Install dependencies into this exact prefix, or submit with "
        "CONDA_ENV_PREFIX=/full/path/to/the/env sbatch scripts/slurm/run_llava_eval.sbatch"
    )
PY

"${PYTHON_BIN}" scripts/hpc/check_nvidia_cuda.py

CMD=("${PYTHON_BIN}" code/test.py --model-path "${MODEL_PATH}" --output-path "${OUTPUT_PATH}")
if [[ -n "${LIMIT}" ]]; then
  CMD+=(--limit "${LIMIT}")
fi

case "${MODEL_PATH}" in
  /*) RESOLVED_MODEL_PATH="${MODEL_PATH}" ;;
  ./*|../*|model/*|models/*) RESOLVED_MODEL_PATH="$(pwd)/${MODEL_PATH}" ;;
  *) RESOLVED_MODEL_PATH="${MODEL_PATH}" ;;
esac
case "${OUTPUT_PATH}" in
  /*) RESOLVED_OUTPUT_PATH="${OUTPUT_PATH}" ;;
  *) RESOLVED_OUTPUT_PATH="$(pwd)/${OUTPUT_PATH}" ;;
esac

echo "MODEL_PATH: ${MODEL_PATH}"
echo "Resolved MODEL_PATH: ${RESOLVED_MODEL_PATH}"
echo "OUTPUT_PATH: ${OUTPUT_PATH}"
echo "Resolved OUTPUT_PATH: ${RESOLVED_OUTPUT_PATH}"
echo "LIMIT: ${LIMIT:-unset}"

"${CMD[@]}"

echo "End: $(date)"
