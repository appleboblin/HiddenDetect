#!/bin/bash
set -euo pipefail

ENV_NAME="${CONDA_ENV:-llava}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
LLAVA_REPO="${LLAVA_REPO:-https://github.com/haotian-liu/LLaVA.git}"
LLAVA_DIR="${LLAVA_DIR:-src/llava}"
LLAVA_COMMIT="${LLAVA_COMMIT:-c121f0432da27facab705978f83c4ada465e46fd}"
RUN_CUDA_CHECK="${RUN_CUDA_CHECK:-0}"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda is not available. Load your cluster's conda/miniconda module first." >&2
  exit 1
fi

eval "$(conda shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi

if [[ ! -d "${LLAVA_DIR}" ]]; then
  mkdir -p "$(dirname "${LLAVA_DIR}")"
  git clone "${LLAVA_REPO}" "${LLAVA_DIR}"
fi

if [[ ! -d "${LLAVA_DIR}/.git" ]]; then
  echo "ERROR: ${LLAVA_DIR} exists but is not a Git checkout. Move it aside or set LLAVA_DIR." >&2
  exit 1
fi

if git -C "${LLAVA_DIR}" cat-file -e "${LLAVA_COMMIT}^{commit}" 2>/dev/null; then
  git -C "${LLAVA_DIR}" checkout "${LLAVA_COMMIT}"
else
  git -C "${LLAVA_DIR}" fetch --tags origin
  git -C "${LLAVA_DIR}" checkout "${LLAVA_COMMIT}"
fi

conda activate "${ENV_NAME}"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e "${LLAVA_DIR}"
python -m pip install -r requirements.txt

if [[ "${RUN_CUDA_CHECK}" == "1" ]]; then
  python scripts/hpc/check_nvidia_cuda.py
fi
