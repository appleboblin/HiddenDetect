#!/bin/bash
set -euo pipefail

ENV_NAME="${CONDA_ENV:-hiddendetect-nvidia}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda is not available. Load your cluster's conda/miniconda module first." >&2
  exit 1
fi

eval "$(conda shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi

conda activate "${ENV_NAME}"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python scripts/hpc/check_nvidia_cuda.py
