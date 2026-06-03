#!/bin/bash

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-model/llava-v1.6-vicuna-7b}"
LIMIT="${LIMIT:-}"
N_FOLDS="${N_FOLDS:-5}"
LAYER_START="${LAYER_START:-16}"
LAYER_END="${LAYER_END:-29}"
RESULTS_DIR="${RESULTS_DIR:-results/llava-comparison-table}"
SUMMARY_PATH="${SUMMARY_PATH:-results/llava-comparison-summary.csv}"
SUMMARY_PYTHON="${SUMMARY_PYTHON:-python}"

SBATCH_ARGS_ARRAY=()
if [[ -n "${SBATCH_ARGS:-}" ]]; then
  read -r -a SBATCH_ARGS_ARRAY <<< "${SBATCH_ARGS}"
fi

SUMMARY_SBATCH_ARGS_ARRAY=()
if [[ -n "${SUMMARY_SBATCH_ARGS:-}" ]]; then
  read -r -a SUMMARY_SBATCH_ARGS_ARRAY <<< "${SUMMARY_SBATCH_ARGS}"
fi

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs "${RESULTS_DIR}" "$(dirname "${SUMMARY_PATH}")"

EXPERIMENTS=(
  paper-default
  fisher-1e-8-all-layers
  fisher-1e-8-paper-layers
  logreg-c-1-all-layers
  logreg-c-0.5-all-layers
)
SCORING_MODES=(trapz fisher fisher logreg logreg)
FISHER_EPSILONS=(1e-8 1e-8 1e-8 1e-8 1e-8)
LOGREG_CS=(0.5 0.5 0.5 1 0.5)
SUPERVISED_LAYER_SCOPES=(selected all selected all all)

JOB_IDS=()

for index in "${!EXPERIMENTS[@]}"; do
  experiment="${EXPERIMENTS[$index]}"
  output_path="${RESULTS_DIR}/${experiment}.csv"
  job_id=$(
    sbatch "${SBATCH_ARGS_ARRAY[@]}" \
      --parsable \
      --export=ALL,MODEL_PATH="${MODEL_PATH}",OUTPUT_PATH="${output_path}",LIMIT="${LIMIT}",SCORING_MODE="${SCORING_MODES[$index]}",N_FOLDS="${N_FOLDS}",FISHER_EPSILON="${FISHER_EPSILONS[$index]}",LOGREG_C="${LOGREG_CS[$index]}",LAYER_START="${LAYER_START}",LAYER_END="${LAYER_END}",SUPERVISED_LAYER_SCOPE="${SUPERVISED_LAYER_SCOPES[$index]}" \
      scripts/slurm/run_llava_eval.sbatch
  )
  JOB_IDS+=("${job_id}")
  echo "Submitted ${experiment}: job ${job_id}, output ${output_path}"
done

dependency_ids=$(IFS=:; echo "${JOB_IDS[*]}")
summary_job_id=$(
  sbatch "${SUMMARY_SBATCH_ARGS_ARRAY[@]}" \
    --parsable \
    --dependency="afterok:${dependency_ids}" \
    --job-name=hiddendetect-llava-summary \
    --output=logs/%x-%j.out \
    --error=logs/%x-%j.err \
    <<SBATCH
#!/bin/bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
"${SUMMARY_PYTHON}" scripts/summarize_eval_results.py \\
  --output "${SUMMARY_PATH}" \\
  --result "paper-default=${RESULTS_DIR}/paper-default.csv" \\
  --result "fisher-1e-8-all-layers=${RESULTS_DIR}/fisher-1e-8-all-layers.csv" \\
  --result "fisher-1e-8-paper-layers=${RESULTS_DIR}/fisher-1e-8-paper-layers.csv" \\
  --result "logreg-c-1-all-layers=${RESULTS_DIR}/logreg-c-1-all-layers.csv" \\
  --result "logreg-c-0.5-all-layers=${RESULTS_DIR}/logreg-c-0.5-all-layers.csv"
SBATCH
)

echo "Submitted summary job ${summary_job_id} after ${dependency_ids}"
echo "Summary path: ${SUMMARY_PATH}"
