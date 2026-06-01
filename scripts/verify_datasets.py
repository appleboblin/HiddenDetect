#!/usr/bin/env python3
"""Verify that the local HiddenDetect datasets are present and readable.

Run from the repository root:

    python scripts/verify_datasets.py

The checks mirror the paths consumed by code/load_datasets.py and the dataset
layout produced by scripts/download_datasets.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data"

MM_SAFETY_CATEGORIES = [
    "Illegal_Activitiy",
    "Physical_Harm",
    "Sex",
    "HateSpeech",
    "Fraud",
    "Malware_Generation",
    "EconomicHarm",
    "Privacy_Violence",
]
MM_SAFETY_SPLITS = ["SD_TYPO.parquet", "SD.parquet", "TYPO.parquet", "Text_only.parquet"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def limited_messages(messages: list[str], max_count: int) -> tuple[list[str], int]:
    if max_count < 0 or len(messages) <= max_count:
        return messages, 0
    return messages[:max_count], len(messages) - max_count


class DatasetVerifier:
    def __init__(self, data_dir: Path, *, check_parquet_contents: bool = True) -> None:
        self.data_dir = data_dir
        self.check_parquet_contents = check_parquet_contents
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.summaries: list[str] = []

    @property
    def error_count(self) -> int:
        return len(self.errors)

    def rel(self, path: Path) -> str:
        try:
            return f"data/{path.resolve().relative_to(self.data_dir.resolve())}"
        except ValueError:
            pass
        try:
            return str(path.resolve().relative_to(REPO_ROOT))
        except ValueError:
            return str(path)

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def require_file(self, path: Path) -> bool:
        if not path.exists():
            self.error(f"missing file: {self.rel(path)}")
            return False
        if not path.is_file():
            self.error(f"not a file: {self.rel(path)}")
            return False
        if path.stat().st_size == 0:
            self.error(f"empty file: {self.rel(path)}")
            return False
        return True

    def require_dir(self, path: Path) -> bool:
        if not path.exists():
            self.error(f"missing directory: {self.rel(path)}")
            return False
        if not path.is_dir():
            self.error(f"not a directory: {self.rel(path)}")
            return False
        return True

    def read_csv_rows(self, path: Path, required_columns: Iterable[str]) -> list[dict[str, str]]:
        if not self.require_file(path):
            return []

        try:
            with path.open(newline="", encoding="utf-8-sig") as csvfile:
                reader = csv.DictReader(csvfile)
                columns = set(reader.fieldnames or [])
                missing = sorted(set(required_columns) - columns)
                if missing:
                    self.error(f"{self.rel(path)} missing columns: {', '.join(missing)}")
                    return []
                rows = list(reader)
        except (csv.Error, UnicodeDecodeError, OSError) as exc:
            self.error(f"could not read CSV {self.rel(path)}: {exc}")
            return []

        if not rows:
            self.error(f"CSV has no data rows: {self.rel(path)}")
        return rows

    def read_csv_min_columns(self, path: Path, min_columns: int) -> list[list[str]]:
        if not self.require_file(path):
            return []

        try:
            with path.open(newline="", encoding="utf-8-sig") as csvfile:
                reader = csv.reader(csvfile)
                header = next(reader, [])
                rows = list(reader)
        except (csv.Error, UnicodeDecodeError, OSError) as exc:
            self.error(f"could not read CSV {self.rel(path)}: {exc}")
            return []

        if len(header) < min_columns:
            self.error(f"{self.rel(path)} needs at least {min_columns} columns")
            return []
        short_rows = [index for index, row in enumerate(rows, start=2) if len(row) < min_columns]
        if short_rows:
            shown = ", ".join(str(index) for index in short_rows[:5])
            self.error(f"{self.rel(path)} has rows with fewer than {min_columns} columns: {shown}")
            return []
        if not rows:
            self.error(f"CSV has no data rows: {self.rel(path)}")
        return rows

    def check_json_list(self, path: Path, required_columns: Iterable[str]) -> list[dict[str, object]]:
        if not self.require_file(path):
            return []

        try:
            with path.open(encoding="utf-8") as jsonfile:
                data = json.load(jsonfile)
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
            self.error(f"could not read JSON {self.rel(path)}: {exc}")
            return []

        if not isinstance(data, list) or not data:
            self.error(f"JSON must contain a non-empty list: {self.rel(path)}")
            return []

        missing_fields = [
            index
            for index, item in enumerate(data)
            if not isinstance(item, dict) or any(field not in item for field in required_columns)
        ]
        if missing_fields:
            shown = ", ".join(str(index) for index in missing_fields[:5])
            self.error(f"{self.rel(path)} entries missing required fields at indexes: {shown}")
            return []

        return data

    def check_parquet(self, path: Path, required_columns: Iterable[str]) -> int:
        if not self.require_file(path):
            return 0
        if not self.check_parquet_contents:
            return 0

        try:
            import pandas as pd
        except ImportError:
            self.warn("pandas is not installed; skipped parquet content checks")
            self.check_parquet_contents = False
            return 0

        try:
            df = pd.read_parquet(path)
        except Exception as exc:  # pandas raises backend-specific exceptions
            self.error(f"could not read parquet {self.rel(path)}: {exc}")
            return 0

        missing = sorted(set(required_columns) - set(df.columns))
        if missing:
            self.error(f"{self.rel(path)} missing columns: {', '.join(missing)}")
            return 0
        if df.empty:
            self.error(f"parquet has no rows: {self.rel(path)}")
            return 0
        return len(df)

    def image_files(self, path: Path) -> list[Path]:
        if not self.require_dir(path):
            return []
        return sorted(file for file in path.rglob("*") if file.suffix.lower() in IMAGE_SUFFIXES)

    def require_referenced_files(self, base_dir: Path, relative_paths: Iterable[str]) -> int:
        missing = 0
        for relative_path in sorted(set(relative_paths)):
            candidate = base_dir / relative_path
            if not self.require_file(candidate):
                missing += 1
        return missing

    def data_relative_reference(self, reference: str) -> str:
        path = Path(reference)
        if path.parts[:1] == ("data",):
            return str(Path(*path.parts[1:]))
        return reference

    def check_xstest(self) -> None:
        path = self.data_dir / "xstest-v2-copy" / "data" / "gpt4-00000-of-00001.parquet"
        rows = self.check_parquet(path, ["prompt", "type"])
        self.summaries.append(f"XSTest: {'checked' if rows else 'path checked'}")

    def check_figstep(self) -> None:
        benign_rows = self.read_csv_min_columns(
            self.data_dir / "FigStep" / "benign_questions.csv",
            2,
        )
        safebench_rows = self.read_csv_rows(
            self.data_dir / "FigStep" / "safebench.csv",
            ["instruction"],
        )
        images = self.image_files(self.data_dir / "FigStep" / "FigImg")
        if not images:
            self.error("FigStep image directory contains no image files")
        self.summaries.append(
            f"FigStep: {len(benign_rows)} benign rows, {len(safebench_rows)} unsafe rows, "
            f"{len(images)} images"
        )

    def check_mm_safety_bench(self) -> None:
        total_rows = 0
        for category in MM_SAFETY_CATEGORIES:
            for split in MM_SAFETY_SPLITS:
                required_columns = ["question"] if split == "Text_only.parquet" else ["question", "image"]
                total_rows += self.check_parquet(
                    self.data_dir / "MM-SafetyBench" / category / split,
                    required_columns,
                )
        self.summaries.append(
            f"MM-SafetyBench: {len(MM_SAFETY_CATEGORIES) * len(MM_SAFETY_SPLITS)} parquet files, "
            f"{total_rows} rows checked"
        )

    def check_jailbreakv(self) -> None:
        rows = self.read_csv_rows(
            self.data_dir / "JailBreakV_28K" / "selected_JBV28K.csv",
            ["redteam_query", "jailbreak_query", "image_path"],
        )
        image_paths = [row["image_path"] for row in rows if row.get("image_path")]
        missing = self.require_referenced_files(self.data_dir / "JailBreakV_28K", image_paths)
        self.summaries.append(
            f"JailBreakV-28K: {len(rows)} selected rows, {len(set(image_paths)) - missing} referenced images"
        )

    def check_vae(self) -> None:
        rows = self.read_csv_rows(self.data_dir / "VAE" / "manual_harmful_instructions.csv", [])
        images = self.image_files(self.data_dir / "VAE" / "Adversarial_Img")
        if not images:
            self.error("VAE adversarial image directory contains no image files")
        self.summaries.append(f"VAE: {len(rows)} harmful prompts, {len(images)} images")

    def check_mm_vet(self) -> None:
        rows = self.check_json_list(
            self.data_dir / "MM-Vet" / "mm-vet_metadata.json",
            ["id", "txt", "img", "toxicity"],
        )
        image_paths = [
            self.data_relative_reference(str(row["img"]))
            for row in rows
            if isinstance(row.get("img"), str) and str(row.get("img"))
        ]
        missing = self.require_referenced_files(self.data_dir, image_paths)
        self.summaries.append(
            f"MM-Vet: {len(rows)} metadata rows, {len(set(image_paths)) - missing} referenced images"
        )

    def check_few_shot(self) -> None:
        zip_path = self.data_dir / "few_shot.zip"
        extracted = self.data_dir / "few_shot"
        if zip_path.exists():
            self.require_file(zip_path)
        elif not extracted.exists():
            self.warn("optional data/few_shot.zip is absent")
        if extracted.exists():
            files = [path for path in extracted.rglob("*") if path.is_file()]
            if not files:
                self.error("data/few_shot exists but contains no files")
            self.summaries.append(f"few_shot: {len(files)} extracted files")
        else:
            self.summaries.append("few_shot: not extracted")

    def check_all(self) -> None:
        if not self.require_dir(self.data_dir):
            return
        self.check_xstest()
        self.check_figstep()
        self.check_mm_safety_bench()
        self.check_jailbreakv()
        self.check_vae()
        self.check_mm_vet()
        self.check_few_shot()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify that HiddenDetect datasets in data/ are present and readable."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="dataset root to verify; defaults to ./data",
    )
    parser.add_argument(
        "--skip-parquet-read",
        action="store_true",
        help="only check parquet files exist and are non-empty; do not import pandas/read contents",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=50,
        help="maximum individual errors to print; use -1 to print all",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    verifier = DatasetVerifier(
        args.data_dir.resolve(),
        check_parquet_contents=not args.skip_parquet_read,
    )
    verifier.check_all()

    for summary in verifier.summaries:
        print(f"[verify-datasets] {summary}")
    for warning in verifier.warnings:
        print(f"[verify-datasets] WARNING: {warning}", file=sys.stderr)
    shown_errors, omitted_errors = limited_messages(verifier.errors, args.max_errors)
    for error in shown_errors:
        print(f"[verify-datasets] ERROR: {error}", file=sys.stderr)
    if omitted_errors:
        print(f"[verify-datasets] ... omitted {omitted_errors} more error(s)", file=sys.stderr)

    if verifier.errors:
        print(f"[verify-datasets] failed with {len(verifier.errors)} error(s)", file=sys.stderr)
        return 1

    print("[verify-datasets] verification passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
