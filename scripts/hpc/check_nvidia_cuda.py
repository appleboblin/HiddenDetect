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


def _device_capability(index: int, props: object) -> str:
    if hasattr(props, "major") and hasattr(props, "minor"):
        return f"{props.major}.{props.minor}"
    if hasattr(props, "major_minor"):
        major_minor = getattr(props, "major_minor")
        if isinstance(major_minor, (tuple, list)) and len(major_minor) == 2:
            return f"{major_minor[0]}.{major_minor[1]}"
    major, minor = torch.cuda.get_device_capability(index)
    return f"{major}.{minor}"


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
        capability = _device_capability(index, props)
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
