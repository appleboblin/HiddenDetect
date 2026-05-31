#!/bin/bash
#SBATCH -J hiddendetect-qwen
#SBATCH -p dgxh
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --mail-type=FAIL,END

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-llava}"
MODEL_PATH="${MODEL_PATH:-model/Qwen-VL-Chat}"
OUTPUT_PATH="${OUTPUT_PATH:-results/qwen-result.csv}"
LIMIT="${LIMIT:-}"

mkdir -p logs results

module load conda
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

python scripts/hpc/check_nvidia_cuda.py

CMD=(python code/test_qwen.py --model-path "${MODEL_PATH}" --output-path "${OUTPUT_PATH}")
if [[ -n "${LIMIT}" ]]; then
  CMD+=(--limit "${LIMIT}")
fi

"${CMD[@]}"
