#!/bin/bash
#SBATCH -J hiddendetect-llava-eps
#SBATCH -p dgxh
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --array=0-3
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
#SBATCH --mail-type=FAIL,END

set -euo pipefail

EPSILON_VALUES=(${FISHER_EPSILONS:-1e-8 1e-9 1e-10 1e-12})
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

if (( TASK_ID < 0 || TASK_ID >= ${#EPSILON_VALUES[@]} )); then
  echo "ERROR: SLURM_ARRAY_TASK_ID=${TASK_ID} is outside FISHER_EPSILONS length ${#EPSILON_VALUES[@]}." >&2
  echo "Submit with a matching array range, for example: sbatch --array=0-3 $0" >&2
  exit 1
fi

export FISHER_EPSILON="${EPSILON_VALUES[$TASK_ID]}"
export OUTPUT_PATH="${OUTPUT_PATH:-results/llava-fisher-epsilon-${FISHER_EPSILON}.csv}"

echo "Sweep task: ${TASK_ID}"
echo "FISHER_EPSILON: ${FISHER_EPSILON}"
echo "OUTPUT_PATH: ${OUTPUT_PATH}"

bash scripts/slurm/run_llava_eval.sbatch
