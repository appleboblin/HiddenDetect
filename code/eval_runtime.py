import csv
import os
from pathlib import Path


CSV_HEADER = ["Dataset Name", "AUPRC", "AUROC"]
LOCAL_MODEL_DIR_NAMES = {"model", "models"}


def _looks_like_local_path(model_path: str, resolved_path: Path) -> bool:
    path = Path(model_path).expanduser()
    parts = path.parts
    return (
        path.is_absolute()
        or model_path.startswith(".")
        or model_path.startswith("~")
        or resolved_path.exists()
        or (bool(parts) and parts[0] in LOCAL_MODEL_DIR_NAMES)
    )


def validate_model_path(model_path: str, repo_root: Path | None = None) -> str:
    """Return a resolved local model path, or the original remote model id."""
    root = Path.cwd() if repo_root is None else Path(repo_root)
    expanded_path = Path(model_path).expanduser()
    resolved_path = expanded_path if expanded_path.is_absolute() else root / expanded_path
    resolved_path = resolved_path.resolve()

    if not _looks_like_local_path(model_path, resolved_path):
        return model_path

    if not resolved_path.exists():
        raise SystemExit(f"ERROR: Model path does not exist: {resolved_path}")
    if not resolved_path.is_dir():
        raise SystemExit(f"ERROR: Model path is not a directory: {resolved_path}")

    config_path = resolved_path / "config.json"
    if not config_path.is_file():
        raise SystemExit(
            "ERROR: Local model path is missing config.json: "
            f"{config_path}. Empty or interrupted model downloads must be restaged."
        )

    return str(resolved_path)


def finish_evaluation(output_path: str | os.PathLike[str], results, failed_datasets) -> None:
    if not results:
        failed_summary = ""
        if failed_datasets:
            failed_summary = f" Failed datasets: {', '.join(failed_datasets)}."
        raise SystemExit(
            "ERROR: No datasets completed successfully; refusing to write "
            f"a header-only result CSV.{failed_summary}"
        )

    output_path = Path(output_path)
    try:
        if output_path.parent != Path("."):
            output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(CSV_HEADER)
            for dataset_name, result in results.items():
                if result is not None:
                    writer.writerow([dataset_name, f"{result[0]:.4f}", f"{result[1]:.4f}"])
    except OSError as exc:
        raise SystemExit(f"ERROR: Error writing to CSV {output_path}: {exc}") from exc

    print(f"Results successfully written to {output_path}")
    if failed_datasets:
        print(f"Failed datasets ({len(failed_datasets)}): {', '.join(failed_datasets)}")
