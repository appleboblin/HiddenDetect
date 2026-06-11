#!/bin/bash
#SBATCH -J hiddendetect-llava-logreg-c
#SBATCH -p dgxh
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --array=0-1
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
#SBATCH --mail-type=FAIL,END

set -euo pipefail

C_VALUES=(${LOGREG_C_VALUES:-0.5 1})
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

if (( TASK_ID < 0 || TASK_ID >= ${#C_VALUES[@]} )); then
  echo "ERROR: SLURM_ARRAY_TASK_ID=${TASK_ID} is outside LOGREG_C_VALUES length ${#C_VALUES[@]}." >&2
  echo "Submit with a matching array range, for example: sbatch --array=0-1 $0" >&2
  exit 1
fi

export SCORING_MODE="logreg"
export LOGREG_C="${C_VALUES[$TASK_ID]}"
export SUPERVISED_LAYER_SCOPE="${SUPERVISED_LAYER_SCOPE:-all}"
export OUTPUT_PATH="${OUTPUT_PATH:-results/llava-logreg-c-${LOGREG_C}.csv}"

echo "Sweep task: ${TASK_ID}"
echo "SCORING_MODE: ${SCORING_MODE}"
echo "LOGREG_C: ${LOGREG_C}"
echo "SUPERVISED_LAYER_SCOPE: ${SUPERVISED_LAYER_SCOPE}"
echo "OUTPUT_PATH: ${OUTPUT_PATH}"

bash scripts/slurm/run_llava_eval.sbatch
