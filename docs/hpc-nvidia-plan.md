# NVIDIA HPC Support Plan

> **For agentic workers:** Implement this plan task by task on branch `hpc-nvidia`. Keep the existing AMD/local runnable path intact while adding a verified NVIDIA CUDA path for HPC.

**Goal:** Make HiddenDetect runnable on an HPC cluster with NVIDIA GPUs while preserving the repository's existing runnable behavior.

**Architecture:** Add a documented Conda-plus-pip HPC workflow, SLURM batch entrypoints, a CUDA environment check, and small runtime-configuration changes to the existing evaluation scripts. The implementation should avoid vendoring large model files or datasets and should keep hardware-specific behavior behind CLI options, environment variables, or PyTorch device selection.

**Tech Stack:** Python, PyTorch CUDA, LLaVA, Hugging Face Transformers, Conda, pip, SLURM, NVIDIA H100/H200 GPU nodes.

---

## Current Repository Facts

- The starting branch is `po`, currently tracking `origin/po`.
- The NVIDIA work branch is `hpc-nvidia`, created from `po`.
- The tracked runnable entrypoints are `code/test.py`, `code/test_qwen.py`, and `code/safety_aware_layers.py`.
- The current `requirements.txt` already pins PyTorch `2.1.2`, TorchVision `0.16.2`, Triton `2.1.0`, and CUDA 12 NVIDIA wheel dependencies.
- The evaluation code currently assumes CUDA in several places through `.cuda()`, `.to("cuda")`, or `device_map="cuda"`.
- Large model and dataset artifacts are expected under local directories such as `model/` and `data/`, but should not be committed.
- There is an untracked `src/` directory in the workspace. Do not add, delete, or rewrite it unless a later task explicitly chooses to vendor LLaVA.

## Target HPC Runtime

Use Conda plus pip as the first supported runtime.

Recommended NVIDIA target:

- Partition: `dgxh`
- GPU request: `--gres=gpu:1`
- Default GPU constraint: `--constraint=h100`
- Optional GPU constraint: `--constraint=h200`
- Default environment name: `llava`

The implementation should document how to override the partition, account, wall time, model path, output path, and GPU constraint without editing Python code.

## Implementation Tasks

### Task 1: Protect the Branch and Repo Shape

**Files:**
- Modify: `.gitignore`
- Keep untouched unless explicitly required: `src/`

- [ ] Confirm the branch is `hpc-nvidia`.

```bash
git status --short --branch
```

Expected: output starts with `## hpc-nvidia`.

- [ ] Add a `.gitignore` if one does not exist.

The file should ignore runtime artifacts without hiding source files:

```gitignore
__pycache__/
*.py[cod]
.pytest_cache/
.ruff_cache/

.env
.venv/
env/
venv/

model/
models/
results/
logs/
slurm-*.out
*.out
*.err

data/MM-SafetyBench/
data/VAE/
data/xstest-v2-copy/
data/FigStep/FigImg/
data/JailBreakV_28K/llm_transfer_attack/
data/few_shot/
```

- [ ] Verify the untracked `src/` directory is still untracked and untouched.

```bash
git status --short
```

Expected: `?? src/` may still appear, and no `src/` files are staged.

### Task 2: Add NVIDIA CUDA Environment Check

**Files:**
- Create: `scripts/hpc/check_nvidia_cuda.py`

- [ ] Add a small check script that validates Python dependencies and CUDA visibility.

```python
import importlib
import sys

import torch


REQUIRED_IMPORTS = [
    "transformers",
    "accelerate",
    "PIL",
    "pandas",
    "sklearn",
    "llava",
]


def main() -> int:
    print(f"python={sys.version.split()[0]}")
    print(f"torch={torch.__version__}")
    print(f"torch_cuda_build={torch.version.cuda}")
    print(f"cuda_available={torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("ERROR: PyTorch cannot see an NVIDIA CUDA GPU.")
        return 1

    device_count = torch.cuda.device_count()
    print(f"cuda_device_count={device_count}")
    for index in range(device_count):
        props = torch.cuda.get_device_properties(index)
        total_gb = props.total_memory / 1024**3
        capability = ".".join(str(part) for part in props.major_minor)
        print(
            f"cuda_device_{index}={props.name}, "
            f"vram_gb={total_gb:.1f}, capability={capability}"
        )

    for module_name in REQUIRED_IMPORTS:
        importlib.import_module(module_name)
        print(f"import_ok={module_name}")

    x = torch.ones((2, 2), device="cuda")
    y = x @ x
    print(f"cuda_tensor_check={y.sum().item():.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Note: if `props.major_minor` is not available in the installed PyTorch version, replace that line with `capability = ".".join(map(str, torch.cuda.get_device_capability(index)))`.

### Task 3: Add Conda Setup Script

**Files:**
- Create: `scripts/hpc/setup_nvidia_env.sh`

- [ ] Add an environment setup script that can be run on an HPC login node.

```bash
#!/bin/bash
set -euo pipefail

ENV_NAME="${CONDA_ENV:-llava}"
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
```

- [ ] Make the script executable.

```bash
chmod +x scripts/hpc/setup_nvidia_env.sh
```

### Task 4: Make LLaVA Evaluation Configurable

**Files:**
- Modify: `code/test.py`

- [ ] Add CLI arguments:

```text
--model-path, default model/llava-v1.6-vicuna-7b/
--output-path, default result.csv
--limit, default None
--seed, default 539
--device, default cuda if available else cpu
```

- [ ] Replace hard-coded CUDA tensor movement with the selected device.

Expected behavior:

- Model loading still uses `device_map="auto"` by default for CUDA.
- Input tensors use `.to(device)` instead of `.cuda()`.
- Image tensors use `.to(model.device, dtype=torch.float16)` only when `model.device` is valid; otherwise use the selected device.
- `random.seed(args.seed)` and `np.random.seed(args.seed)` run before dataset sampling.
- `--limit N` slices each assembled dataset to at most `N` samples for smoke tests.
- Results are written to `args.output_path`, creating the parent directory when needed.

- [ ] Keep the default full evaluation behavior unchanged when no CLI args are passed.

### Task 5: Make Qwen Evaluation Configurable

**Files:**
- Modify: `code/test_qwen.py`

- [ ] Add CLI arguments matching `code/test.py`:

```text
--model-path, default ./model/Qwen-VL-Chat
--output-path, default result_qwen.csv
--limit, default None
--seed, default 539
--device, default cuda if available else cpu
```

- [ ] Replace hard-coded CUDA assumptions.

Expected behavior:

- Use `device_map="auto"` when the selected device is CUDA.
- Use the selected device for `input_ids`.
- Results are written to `args.output_path`, creating the parent directory when needed.
- `--limit N` supports smoke tests.

### Task 6: Add SLURM Batch Scripts

**Files:**
- Create: `scripts/slurm/run_llava_eval.sbatch`
- Create: `scripts/slurm/run_qwen_eval.sbatch`

- [ ] Add a LLaVA batch script.

```bash
#!/bin/bash
#SBATCH -J hiddendetect-llava
#SBATCH -p dgxh
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-llava}"
MODEL_PATH="${MODEL_PATH:-model/llava-v1.6-vicuna-7b}"
OUTPUT_PATH="${OUTPUT_PATH:-results/llava-result.csv}"
LIMIT="${LIMIT:-}"

mkdir -p logs results

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

python scripts/hpc/check_nvidia_cuda.py

CMD=(python code/test.py --model-path "${MODEL_PATH}" --output-path "${OUTPUT_PATH}")
if [[ -n "${LIMIT}" ]]; then
  CMD+=(--limit "${LIMIT}")
fi

"${CMD[@]}"
```

- [ ] Add a Qwen batch script with the same structure and:

```bash
#SBATCH -J hiddendetect-qwen
MODEL_PATH="${MODEL_PATH:-model/Qwen-VL-Chat}"
OUTPUT_PATH="${OUTPUT_PATH:-results/qwen-result.csv}"
CMD=(python code/test_qwen.py --model-path "${MODEL_PATH}" --output-path "${OUTPUT_PATH}")
```

- [ ] Make both scripts executable.

```bash
chmod +x scripts/slurm/run_llava_eval.sbatch scripts/slurm/run_qwen_eval.sbatch
```

### Task 7: Document User-Facing HPC Usage

**Files:**
- Modify: `README.md`

- [ ] Add a short `HPC NVIDIA GPU` section after the install section.

The section should include:

```bash
git switch hpc-nvidia
bash scripts/hpc/setup_nvidia_env.sh
```

Smoke test:

```bash
LIMIT=1 sbatch scripts/slurm/run_llava_eval.sbatch
```

Full run:

```bash
sbatch scripts/slurm/run_llava_eval.sbatch
```

Optional H200 run:

```bash
sbatch --constraint=h200 scripts/slurm/run_llava_eval.sbatch
```

Mention that model weights must be staged under `model/` or passed with `MODEL_PATH=/path/to/model`.

## Validation Plan

- [ ] Run Python syntax checks.

```bash
python -m py_compile \
  code/test.py \
  code/test_qwen.py \
  code/safety_aware_layers.py \
  code/load_datasets.py \
  scripts/hpc/check_nvidia_cuda.py
```

Expected: command exits with status `0`.

- [ ] On an NVIDIA GPU compute node, validate CUDA and imports.

```bash
python scripts/hpc/check_nvidia_cuda.py
```

Expected:

- `cuda_available=True`
- At least one NVIDIA GPU listed
- `import_ok=llava`
- `cuda_tensor_check=8.0`

- [ ] Submit a LLaVA smoke job.

```bash
LIMIT=1 sbatch scripts/slurm/run_llava_eval.sbatch
```

Expected: SLURM job completes and writes `results/llava-result.csv`.

- [ ] Submit a Qwen smoke job if Qwen model files are available.

```bash
LIMIT=1 sbatch scripts/slurm/run_qwen_eval.sbatch
```

Expected: SLURM job completes and writes `results/qwen-result.csv`.

- [ ] Run the full LLaVA evaluation after smoke tests pass.

```bash
sbatch scripts/slurm/run_llava_eval.sbatch
```

Expected: SLURM job completes and writes the full result CSV.

## Acceptance Criteria

- Branch `hpc-nvidia` exists and is based on `po`.
- `docs/hpc-nvidia-plan.md` exists and describes the NVIDIA HPC work.
- The repo includes a CUDA verification script and SLURM entrypoints.
- `code/test.py` and `code/test_qwen.py` can run with explicit model and output paths.
- Smoke runs can be submitted with `LIMIT=1`.
- Existing local/AMD runnable behavior is not intentionally removed.
- Large models, generated results, logs, and expanded datasets are not committed.
