#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


SUMMARY_HEADER = ["Experiment", "Dataset Name", "AUPRC", "AUROC"]


def _parse_result_spec(result_spec: str) -> tuple[str, Path]:
    if "=" not in result_spec:
        raise argparse.ArgumentTypeError(
            f"Expected EXPERIMENT=CSV_PATH, got {result_spec!r}."
        )
    experiment, csv_path = result_spec.split("=", 1)
    if not experiment:
        raise argparse.ArgumentTypeError("Experiment name cannot be empty.")
    if not csv_path:
        raise argparse.ArgumentTypeError("CSV path cannot be empty.")
    return experiment, Path(csv_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge HiddenDetect evaluation CSVs into a long-form summary."
    )
    parser.add_argument(
        "--result",
        action="append",
        type=_parse_result_spec,
        required=True,
        metavar="EXPERIMENT=CSV_PATH",
        help="Evaluation CSV to include. May be passed multiple times.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Where to write the merged long-form summary CSV.",
    )
    return parser.parse_args()


def _read_result_rows(experiment: str, csv_path: Path) -> list[list[str]]:
    if not csv_path.is_file():
        raise SystemExit(f"ERROR: Missing result CSV for {experiment}: {csv_path}")

    rows = []
    with csv_path.open(newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        missing_columns = [
            column
            for column in ("Dataset Name", "AUPRC", "AUROC")
            if column not in (reader.fieldnames or [])
        ]
        if missing_columns:
            raise SystemExit(
                f"ERROR: {csv_path} is missing columns: {', '.join(missing_columns)}"
            )
        for row in reader:
            rows.append(
                [
                    experiment,
                    row["Dataset Name"],
                    row["AUPRC"],
                    row["AUROC"],
                ]
            )
    return rows


def write_summary(results: list[tuple[str, Path]], output_path: Path) -> None:
    summary_rows = []
    for experiment, csv_path in results:
        summary_rows.extend(_read_result_rows(experiment, csv_path))

    if not summary_rows:
        raise SystemExit("ERROR: No result rows found; refusing to write an empty summary.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(SUMMARY_HEADER)
        writer.writerows(summary_rows)

    print(f"Summary written to {output_path}")


def main() -> None:
    args = parse_args()
    write_summary(args.result, Path(args.output))


if __name__ == "__main__":
    main()
