# HPC NVIDIA End-to-End Runbook

## Prerequisites

- You have access to an NVIDIA GPU SLURM partition (default scripts use `dgxh` and `--constraint=h100`).
- `conda` is available on the cluster. The batch scripts load the `conda` module before activating the environment.
- `sbatch` is available.
- Model weights are staged locally (for example under `model/`), or you know the path to pass via `MODEL_PATH`.
- Evaluation data is staged under `data/` as expected by the evaluation scripts.

## 1) Clone and set up the environment

Repository branch: <https://github.com/appleboblin/HiddenDetect/tree/hpc-nvidia>

```bash
git clone --branch hpc-nvidia https://github.com/appleboblin/HiddenDetect.git
cd HiddenDetect
bash scripts/hpc/setup_nvidia_env.sh
```

The setup script follows the upstream LLaVA install order:

1. Create or reuse one Conda environment named `llava`.
2. Clone LLaVA into `src/llava` if it is missing.
3. Check out the pinned LLaVA commit.
4. Install LLaVA editably.
5. Install HiddenDetect dependencies from `requirements.txt`.

Optional overrides:

```bash
CONDA_ENV=my-env PYTHON_VERSION=3.10 bash scripts/hpc/setup_nvidia_env.sh
LLAVA_COMMIT=<commit-or-ref> bash scripts/hpc/setup_nvidia_env.sh
```

Manual fallback:

```bash
git clone https://github.com/haotian-liu/LLaVA.git src/llava
cd src/llava
git checkout c121f0432da27facab705978f83c4ada465e46fd
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip setuptools wheel
pip install -e .
cd ../..
pip install -r requirements.txt
```

## 2) GPU/dependency verification

Run CUDA verification only inside an interactive GPU allocation or a SLURM job. Login nodes commonly have no visible GPU, so `cuda_available=False` there is expected.

```bash
conda activate llava
python scripts/hpc/check_nvidia_cuda.py
```

If you are already inside a GPU allocation when setting up, you can run the setup and CUDA check together:

```bash
RUN_CUDA_CHECK=1 bash scripts/hpc/setup_nvidia_env.sh
```

Expected markers in output:

- `cuda_available=True`
- At least one `cuda_device_0=...` line
- `import_ok=llava`
- `cuda_tensor_check=8.0`

## 3) Smoke run (LLaVA)

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

## 4) Full run (LLaVA)

```bash
sbatch scripts/slurm/run_llava_eval.sbatch
```

## 5) Optional Qwen run

Smoke:

```bash
LIMIT=1 sbatch scripts/slurm/run_qwen_eval.sbatch
```

Full:

```bash
sbatch scripts/slurm/run_qwen_eval.sbatch
```

## 6) Overrides without code edits

Scheduler/account/time/GPU override:

```bash
sbatch -p <partition> -A <account> -t 04:00:00 --constraint=h200 scripts/slurm/run_llava_eval.sbatch
```

Model/output override:

```bash
MODEL_PATH=/path/to/model OUTPUT_PATH=/path/to/results.csv sbatch scripts/slurm/run_llava_eval.sbatch
```

You can also change the Conda env used by batch jobs by name:

```bash
CONDA_ENV=my-env sbatch scripts/slurm/run_llava_eval.sbatch
```

If the batch job cannot find `torch`, submit with the full environment prefix
so SLURM activates the exact environment that contains the dependencies:

```bash
CONDA_ENV_PREFIX=/full/path/to/the/env sbatch scripts/slurm/run_llava_eval.sbatch
```

## 7) Where results land

- LLaVA batch default: `results/llava-result.csv`
- Qwen batch default: `results/qwen-result.csv`

Example checks:

```bash
ls -lh results/
head -n 5 results/llava-result.csv
head -n 5 results/qwen-result.csv
```

## 8) Troubleshooting

- `conda: command not found`
  - Confirm the cluster provides a `conda` module and that `module load conda` works in the batch environment.
- Interrupted LLaVA editable wheel build
  - Re-run `bash scripts/hpc/setup_nvidia_env.sh`. The script reuses `src/llava`, re-checks the pinned commit, upgrades build tooling, and repeats `pip install -e src/llava`.
- Existing half-created Conda environment
  - Re-run the setup script first. If the environment is broken, remove it with `conda env remove -n llava`, then run `bash scripts/hpc/setup_nvidia_env.sh` again. Use the same name you passed through `CONDA_ENV` if you overrode the default.
- `ModuleNotFoundError: No module named 'torch'` / `Activated Python environment does not contain torch`
  - The batch job activated a Python environment that does not have the project dependencies. Submit with `CONDA_ENV_PREFIX=/full/path/to/the/env sbatch scripts/slurm/run_llava_eval.sbatch`, or install dependencies into the printed `Conda prefix` with `python -m pip install -r requirements.txt`.
- Existing `src/llava` checkout
  - The setup script reuses it and checks out `LLAVA_COMMIT`. If `src/llava` exists but is not a Git checkout, move it aside or set `LLAVA_DIR=/path/to/LLaVA`.
- `cuda_available=False` in CUDA check
  - On login nodes this is expected. Run `python scripts/hpc/check_nvidia_cuda.py` inside an interactive GPU allocation or rely on the SLURM scripts, which run the check after the job receives a GPU.
- Model path errors (`No such file or directory` / model load failure)
  - Confirm model files exist and pass `MODEL_PATH=/absolute/or/repo-relative/path`.
- `sbatch` rejects submission due to account/partition/QOS
  - Re-submit with explicit scheduler flags such as `-A <account> -p <partition>`.
