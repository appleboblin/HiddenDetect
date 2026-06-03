#!/bin/bash

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

C_VALUES=(${LOGREG_C_VALUES:-0.03125 0.0625 0.125 0.25 0.5 1 2 4 8})

if (( ${#C_VALUES[@]} == 0 )); then
  echo "ERROR: LOGREG_C_VALUES cannot be empty." >&2
  exit 1
fi

SBATCH_ARGS_ARRAY=()
if [[ -n "${SBATCH_ARGS:-}" ]]; then
  read -r -a SBATCH_ARGS_ARRAY <<< "${SBATCH_ARGS}"
fi

array_end=$((${#C_VALUES[@]} - 1))
array_range="0-${array_end}"
logreg_c_values="${C_VALUES[*]}"

mkdir -p logs results

job_id=$(
  sbatch "${SBATCH_ARGS_ARRAY[@]}" \
    --parsable \
    --array="${array_range}" \
    --export=ALL,LOGREG_C_VALUES="${logreg_c_values}" \
    scripts/slurm/run_llava_logreg_c_sweep.sbatch
)

echo "Submitted LogReg C wide sweep: job ${job_id}"
echo "LOGREG_C_VALUES: ${logreg_c_values}"
echo "Array range: ${array_range}"
echo "Result pattern: results/llava-logreg-c-<C>.csv"
