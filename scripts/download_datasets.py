#!/usr/bin/env python3
"""Download the datasets expected by HiddenDetect into ./data.

Run from the repository root:

    python scripts/download_datasets.py

The script is intentionally resumable: existing files are skipped unless
--overwrite is provided.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable

try:
    from scripts.verify_datasets import DatasetVerifier, limited_messages
except ModuleNotFoundError:  # pragma: no cover - used when executed as a script
    from verify_datasets import DatasetVerifier, limited_messages


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
TMP_DIR = REPO_ROOT / ".dataset_downloads"

FIGSTEP_REPO = "CryptoAILab/FigStep"
VAE_REPO = "Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models"


def log(message: str) -> None:
    print(f"[download-datasets] {message}", flush=True)


def request_headers() -> dict[str, str]:
    headers = {"User-Agent": "HiddenDetect dataset downloader"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def require_huggingface_hub():
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as exc:  # pragma: no cover - user environment check
        raise SystemExit(
            "Missing dependency: huggingface_hub. Install project requirements first:\n"
            "  pip install -r requirements.txt"
        ) from exc
    return hf_hub_download, snapshot_download


def require_hf_datasets():
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - user environment check
        raise SystemExit(
            "Missing dependency: datasets. Install project requirements first:\n"
            "  pip install -r requirements.txt"
        ) from exc
    return load_dataset


def download_url(url: str, dest: Path, overwrite: bool) -> None:
    if dest.exists() and not overwrite:
        log(f"skip existing {dest.relative_to(REPO_ROOT)}")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers=request_headers())
    log(f"download {url} -> {dest.relative_to(REPO_ROOT)}")
    with urllib.request.urlopen(req) as response, tmp.open("wb") as out:
        shutil.copyfileobj(response, out)
    tmp.replace(dest)


def github_api_json(url: str) -> object:
    req = urllib.request.Request(url, headers=request_headers())
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode("utf-8"))


def github_tree_files(repo: str, subdir: str) -> list[dict[str, str]]:
    url = f"https://api.github.com/repos/{repo}/contents/{subdir}"
    entries = github_api_json(url)
    if not isinstance(entries, list):
        raise RuntimeError(f"Unexpected GitHub API response for {repo}/{subdir}")

    files: list[dict[str, str]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        entry_type = entry.get("type")
        entry_path = entry.get("path")
        if entry_type == "file":
            files.append(entry)
        elif entry_type == "dir" and isinstance(entry_path, str):
            files.extend(github_tree_files(repo, entry_path))
    return files


def github_file_download_url(repo: str, path: str) -> str:
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    entry = github_api_json(url)
    if not isinstance(entry, dict) or not entry.get("download_url"):
        raise RuntimeError(f"Could not resolve GitHub file download URL for {repo}/{path}")
    return str(entry["download_url"])


def flattened_name(entry_path: str, root: str) -> str:
    relative = Path(entry_path).relative_to(root)
    return "__".join(relative.parts)


def copy_tree_contents(src: Path, dest: Path, overwrite: bool) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Expected downloaded directory does not exist: {src}")

    for source_path in src.rglob("*"):
        if not source_path.is_file():
            continue
        relative = source_path.relative_to(src)
        target = dest / relative
        if target.exists() and not overwrite:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target)


def download_hf_snapshot(
    repo_id: str,
    target_dir: Path,
    allow_patterns: list[str],
    overwrite: bool,
) -> Path:
    _, snapshot_download = require_huggingface_hub()
    target_dir.mkdir(parents=True, exist_ok=True)
    log(f"download Hugging Face dataset {repo_id}")
    return Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(target_dir),
            allow_patterns=allow_patterns,
            token=os.environ.get("HF_TOKEN"),
        )
    )


def download_xstest(overwrite: bool) -> None:
    download_hf_snapshot(
        "natolambert/xstest-v2-copy",
        DATA_DIR / "xstest-v2-copy",
        ["data/gpt4-00000-of-00001.parquet"],
        overwrite,
    )


def download_mm_safety_bench(overwrite: bool) -> None:
    temp_target = TMP_DIR / "MM-SafetyBench"
    snapshot = download_hf_snapshot(
        "PKU-Alignment/MM-SafetyBench",
        temp_target,
        ["data/*/*.parquet"],
        overwrite,
    )
    copy_tree_contents(snapshot / "data", DATA_DIR / "MM-SafetyBench", overwrite)


def download_figstep(overwrite: bool) -> None:
    download_url(
        github_file_download_url(FIGSTEP_REPO, "data/question/safebench.csv"),
        DATA_DIR / "FigStep" / "safebench.csv",
        overwrite,
    )

    files = github_tree_files(FIGSTEP_REPO, "data/images/SafeBench")
    image_files = [entry for entry in files if str(entry.get("name", "")).lower().endswith(".png")]
    if not image_files:
        raise RuntimeError("No FigStep SafeBench PNG files found on GitHub")

    target_dir = DATA_DIR / "FigStep" / "FigImg"
    for entry in image_files:
        name = flattened_name(str(entry["path"]), "data/images/SafeBench")
        download_url(str(entry["download_url"]), target_dir / name, overwrite)


def selected_jailbreakv_image_paths() -> list[str]:
    csv_path = DATA_DIR / "JailBreakV_28K" / "selected_JBV28K.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            "Missing data/JailBreakV_28K/selected_JBV28K.csv. "
            "This project uses a curated subset CSV that should stay with the repo."
        )

    with csv_path.open(newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        paths = sorted({row["image_path"] for row in reader if row.get("image_path")})
    if not paths:
        raise RuntimeError("No image_path values found in selected_JBV28K.csv")
    return paths


def download_jailbreakv(overwrite: bool, all_images: bool) -> None:
    if all_images:
        temp_target = TMP_DIR / "JailBreakV-28K"
        snapshot = download_hf_snapshot(
            "JailbreakV-28K/JailBreakV-28k",
            temp_target,
            ["JailBreakV_28K/llm_transfer_attack/*.png"],
            overwrite,
        )
        copy_tree_contents(
            snapshot / "JailBreakV_28K" / "llm_transfer_attack",
            DATA_DIR / "JailBreakV_28K" / "llm_transfer_attack",
            overwrite,
        )
        return

    log("download JailBreakV images referenced by selected_JBV28K.csv")
    hf_hub_download, _ = require_huggingface_hub()
    try:
        from huggingface_hub.errors import EntryNotFoundError
    except ImportError:  # pragma: no cover - compatibility with older huggingface_hub
        EntryNotFoundError = RuntimeError  # type: ignore[assignment]

    unavailable: list[str] = []
    for image_path in selected_jailbreakv_image_paths():
        target = DATA_DIR / "JailBreakV_28K" / image_path
        if target.exists() and not overwrite:
            continue
        try:
            downloaded = hf_hub_download(
                repo_id="JailbreakV-28K/JailBreakV-28k",
                repo_type="dataset",
                filename=f"JailBreakV_28K/{image_path}",
                token=os.environ.get("HF_TOKEN"),
            )
        except EntryNotFoundError:
            unavailable.append(image_path)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(downloaded, target)
    if unavailable:
        log(
            "WARNING: JailBreakV source is missing "
            f"{len(unavailable)} selected image(s); final verification will fail"
        )


def download_vae(overwrite: bool) -> None:
    download_url(
        github_file_download_url(VAE_REPO, "harmful_corpus/manual_harmful_instructions.csv"),
        DATA_DIR / "VAE" / "manual_harmful_instructions.csv",
        overwrite,
    )

    files = github_tree_files(VAE_REPO, "adversarial_images")
    image_files = [
        entry
        for entry in files
        if str(entry.get("name", "")).lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    if not image_files:
        raise RuntimeError("No VAE adversarial image files found on GitHub")

    target_dir = DATA_DIR / "VAE" / "Adversarial_Img"
    for entry in image_files:
        download_url(str(entry["download_url"]), target_dir / str(entry["name"]), overwrite)


def mm_vet_question_id(row: object) -> str:
    try:
        question_id = row["question_id"]  # type: ignore[index]
    except (KeyError, TypeError) as exc:
        raise RuntimeError("MM-Vet row is missing question_id") from exc
    if not isinstance(question_id, str) or not question_id:
        raise RuntimeError(f"Invalid MM-Vet question_id: {question_id!r}")
    return question_id


def write_mm_vet_image(image: object, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(image, "save"):
        image_to_save = image
        if getattr(image, "mode", None) == "CMYK" and hasattr(image, "convert"):
            image_to_save = image.convert("RGB")  # type: ignore[union-attr]
        image_to_save.save(target, format="PNG")  # type: ignore[union-attr]
        return

    if isinstance(image, dict):
        if isinstance(image.get("bytes"), bytes):
            target.write_bytes(image["bytes"])
            return
        if image.get("path"):
            shutil.copy2(str(image["path"]), target)
            return

    if isinstance(image, (bytes, bytearray)):
        target.write_bytes(bytes(image))
        return

    raise RuntimeError(f"Unsupported MM-Vet image value for {target.name}: {type(image).__name__}")


def write_mm_vet_images(rows: Iterable[object], target_dir: Path, overwrite: bool) -> int:
    written = 0
    for row in rows:
        question_id = mm_vet_question_id(row)
        try:
            image = row["image"]  # type: ignore[index]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(f"MM-Vet row {question_id} is missing image") from exc

        target = target_dir / f"{question_id}.png"
        if target.exists() and not overwrite:
            continue
        write_mm_vet_image(image, target)
        written += 1
    return written


def download_mm_vet(overwrite: bool) -> None:
    load_dataset = require_hf_datasets()
    log("download Hugging Face dataset lmms-lab/MMVet")
    dataset = load_dataset(
        "lmms-lab/MMVet",
        split="test",
        token=os.environ.get("HF_TOKEN"),
    )
    written = write_mm_vet_images(dataset, DATA_DIR / "MM-Vet", overwrite)
    log(f"MM-Vet images written: {written}")


def unzip_few_shot(overwrite: bool) -> None:
    zip_path = DATA_DIR / "few_shot.zip"
    target_dir = DATA_DIR / "few_shot"
    if not zip_path.exists():
        log("skip few_shot: data/few_shot.zip is not present")
        return
    if target_dir.exists() and not overwrite:
        log("skip existing data/few_shot")
        return
    log("extract data/few_shot.zip")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(DATA_DIR)


def verify(_all_jailbreakv: bool) -> None:
    verifier = DatasetVerifier(DATA_DIR)
    verifier.check_all()
    for summary in verifier.summaries:
        log(summary)
    for warning in verifier.warnings:
        log(f"WARNING: {warning}")
    if verifier.errors:
        shown_errors, omitted_errors = limited_messages(verifier.errors, 50)
        formatted = "\n".join(f"  - {error}" for error in shown_errors)
        if omitted_errors:
            formatted = f"{formatted}\n  - ... omitted {omitted_errors} more error(s)"
        raise SystemExit(
            f"Download finished, but dataset verification failed with "
            f"{verifier.error_count} error(s):\n{formatted}"
        )
    log("verification passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download datasets into the local data/ tree expected by HiddenDetect."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace files that already exist",
    )
    parser.add_argument(
        "--jailbreakv-all",
        action="store_true",
        help=(
            "download all JailBreakV llm_transfer_attack images instead of only the "
            "images referenced by data/JailBreakV_28K/selected_JBV28K.csv"
        ),
    )
    parser.add_argument(
        "--skip",
        action="append",
        default=[],
        choices=["xstest", "figstep", "mm-safetybench", "jailbreakv", "vae", "mm-vet", "few-shot"],
        help="dataset stage to skip; may be repeated",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    skip = set(args.skip)

    stages = [
        ("xstest", lambda: download_xstest(args.overwrite)),
        ("figstep", lambda: download_figstep(args.overwrite)),
        ("mm-safetybench", lambda: download_mm_safety_bench(args.overwrite)),
        ("jailbreakv", lambda: download_jailbreakv(args.overwrite, args.jailbreakv_all)),
        ("vae", lambda: download_vae(args.overwrite)),
        ("mm-vet", lambda: download_mm_vet(args.overwrite)),
        ("few-shot", lambda: unzip_few_shot(args.overwrite)),
    ]

    try:
        for name, stage in stages:
            if name in skip:
                log(f"skip stage {name}")
                continue
            log(f"start {name}")
            stage()
        verify(args.jailbreakv_all)
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"HTTP error {exc.code} while downloading {exc.url}: {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"Network error: {exc.reason}") from exc

    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
