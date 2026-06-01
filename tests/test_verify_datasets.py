import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.verify_datasets import DatasetVerifier, limited_messages


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


class DatasetVerifierTests(unittest.TestCase):
    def test_limited_messages_reports_omitted_count(self) -> None:
        shown, omitted = limited_messages(["a", "b", "c"], 2)

        self.assertEqual(shown, ["a", "b"])
        self.assertEqual(omitted, 1)

    def test_figstep_benign_csv_matches_loader_positional_format(self) -> None:
        with TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            write_csv(
                data_dir / "FigStep" / "benign_questions.csv",
                [{"question": "How do I cook rice?", "FigTxt": "0"}],
            )
            write_csv(
                data_dir / "FigStep" / "safebench.csv",
                [{"instruction": "unsafe prompt"}],
            )
            image_path = data_dir / "FigStep" / "FigImg" / "0.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(b"not-empty")

            verifier = DatasetVerifier(data_dir)
            verifier.check_figstep()

            self.assertEqual(verifier.error_count, 0)

    def test_missing_jailbreakv_referenced_image_is_reported(self) -> None:
        with TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            write_csv(
                data_dir / "JailBreakV_28K" / "selected_JBV28K.csv",
                [
                    {
                        "redteam_query": "unsafe",
                        "jailbreak_query": "bypass",
                        "image_path": "llm_transfer_attack/missing.png",
                    }
                ],
            )

            verifier = DatasetVerifier(data_dir)
            verifier.check_jailbreakv()

            self.assertEqual(verifier.error_count, 1)
            self.assertIn(
                "data/JailBreakV_28K/llm_transfer_attack/missing.png",
                verifier.errors[0],
            )

    def test_existing_jailbreakv_referenced_image_passes(self) -> None:
        with TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            image_path = data_dir / "JailBreakV_28K" / "llm_transfer_attack" / "ok.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(b"not-empty")
            write_csv(
                data_dir / "JailBreakV_28K" / "selected_JBV28K.csv",
                [
                    {
                        "redteam_query": "unsafe",
                        "jailbreak_query": "bypass",
                        "image_path": "llm_transfer_attack/ok.png",
                    }
                ],
            )

            verifier = DatasetVerifier(data_dir)
            verifier.check_jailbreakv()

            self.assertEqual(verifier.error_count, 0)

    def test_missing_mm_vet_referenced_image_is_reported(self) -> None:
        with TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            metadata_path = data_dir / "MM-Vet" / "mm-vet_metadata.json"
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            metadata_path.write_text(
                json.dumps(
                    [
                        {
                            "id": "v1_missing",
                            "txt": "What is shown?",
                            "img": "data/MM-Vet/v1_missing.png",
                            "toxicity": 0,
                        }
                    ]
                ),
                encoding="utf-8",
            )

            verifier = DatasetVerifier(data_dir)
            verifier.check_mm_vet()

            self.assertEqual(verifier.error_count, 1)
            self.assertIn("data/MM-Vet/v1_missing.png", verifier.errors[0])

    def test_existing_mm_vet_referenced_image_passes(self) -> None:
        with TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            image_path = data_dir / "MM-Vet" / "v1_ok.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(b"not-empty")
            metadata_path = data_dir / "MM-Vet" / "mm-vet_metadata.json"
            metadata_path.write_text(
                json.dumps(
                    [
                        {
                            "id": "v1_ok",
                            "txt": "What is shown?",
                            "img": "data/MM-Vet/v1_ok.png",
                            "toxicity": 0,
                        }
                    ]
                ),
                encoding="utf-8",
            )

            verifier = DatasetVerifier(data_dir)
            verifier.check_mm_vet()

            self.assertEqual(verifier.error_count, 0)


if __name__ == "__main__":
    unittest.main()
