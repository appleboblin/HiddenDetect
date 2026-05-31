# HPC NVIDIA End-to-End Runbook

## Prerequisites

- You have access to an NVIDIA GPU SLURM partition (default scripts use `dgxh` and `--constraint=h100`).
- `conda` is available on the cluster. The batch scripts load the `conda` module before activating the environment.
- `sbatch` is available.
- Model weights are staged locally (for example under `model/`), or you know the path to pass via `MODEL_PATH`.
- Evaluation data is staged under `data/` as expected by the evaluation scripts.

## 1) Clone and checkout

Repository branch: <https://github.com/appleboblin/HiddenDetect/tree/hpc-nvidia>

```bash
git clone --branch hpc-nvidia https://github.com/appleboblin/HiddenDetect.git
cd HiddenDetect
```

## 2) Environment setup

```bash
bash scripts/hpc/setup_nvidia_env.sh
```

Optional overrides:

```bash
CONDA_ENV=my-hiddendetect-env PYTHON_VERSION=3.10 bash scripts/hpc/setup_nvidia_env.sh
```

## 3) GPU/dependency verification

```bash
python scripts/hpc/check_nvidia_cuda.py
```

Expected markers in output:

- `cuda_available=True`
- At least one `cuda_device_0=...` line
- `import_ok=llava`
- `cuda_tensor_check=8.0`

## 4) Smoke run (LLaVA)

```bash
LIMIT=1 sbatch scripts/slurm/run_llava_eval.sbatch
```

Inspect logs:

```bash
ls -lt logs/ | head
```

Primary SLURM log locations from the script:

- `logs/%x-%j.out`
- `logs/%x-%j.err`

## 5) Full run (LLaVA)

```bash
sbatch scripts/slurm/run_llava_eval.sbatch
```

## 6) Optional Qwen run

Smoke:

```bash
LIMIT=1 sbatch scripts/slurm/run_qwen_eval.sbatch
```

Full:

```bash
sbatch scripts/slurm/run_qwen_eval.sbatch
```

## 7) Overrides without code edits

Scheduler/account/time/GPU override:

```bash
sbatch -p <partition> -A <account> -t 04:00:00 --constraint=h200 scripts/slurm/run_llava_eval.sbatch
```

Model/output override:

```bash
MODEL_PATH=/path/to/model OUTPUT_PATH=/path/to/results.csv sbatch scripts/slurm/run_llava_eval.sbatch
```

You can also change the Conda env used by batch jobs:

```bash
CONDA_ENV=my-hiddendetect-env sbatch scripts/slurm/run_llava_eval.sbatch
```

## 8) Where results land

- LLaVA batch default: `results/llava-result.csv`
- Qwen batch default: `results/qwen-result.csv`

Example checks:

```bash
ls -lh results/
head -n 5 results/llava-result.csv
head -n 5 results/qwen-result.csv
```

## 9) Troubleshooting

- `conda: command not found`
  - Confirm the cluster provides a `conda` module and that `module load conda` works in the batch environment.
- `cuda_available=False` in CUDA check
  - Verify you are on a GPU node/allocation and your PyTorch CUDA build is available in that environment.
- Model path errors (`No such file or directory` / model load failure)
  - Confirm model files exist and pass `MODEL_PATH=/absolute/or/repo-relative/path`.
- `sbatch` rejects submission due to account/partition/QOS
  - Re-submit with explicit scheduler flags such as `-A <account> -p <partition>`.
