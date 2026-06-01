import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from scripts import download_datasets


class FakeImage:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.save_calls = 0

    def save(self, target: Path, format: str) -> None:
        self.save_calls += 1
        target.write_bytes(self.payload + format.encode("ascii"))


class DownloadDatasetsTests(unittest.TestCase):
    def test_write_mm_vet_images_uses_question_id_png_and_respects_overwrite(self) -> None:
        with TemporaryDirectory() as tmpdir:
            target_dir = Path(tmpdir)
            image = FakeImage(b"first")

            written = download_datasets.write_mm_vet_images(
                [{"question_id": "v1_fake", "image": image}],
                target_dir,
                overwrite=False,
            )

            target = target_dir / "v1_fake.png"
            self.assertEqual(written, 1)
            self.assertEqual(target.read_bytes(), b"firstPNG")

            skipped = download_datasets.write_mm_vet_images(
                [{"question_id": "v1_fake", "image": FakeImage(b"second")}],
                target_dir,
                overwrite=False,
            )

            self.assertEqual(skipped, 0)
            self.assertEqual(target.read_bytes(), b"firstPNG")

            overwritten = download_datasets.write_mm_vet_images(
                [{"question_id": "v1_fake", "image": FakeImage(b"third")}],
                target_dir,
                overwrite=True,
            )

            self.assertEqual(overwritten, 1)
            self.assertEqual(target.read_bytes(), b"thirdPNG")

    def test_downloader_verification_reports_missing_jailbreakv_reference(self) -> None:
        with TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            jailbreakv_dir = data_dir / "JailBreakV_28K"
            (jailbreakv_dir / "llm_transfer_attack").mkdir(parents=True)
            (jailbreakv_dir / "selected_JBV28K.csv").write_text(
                "redteam_query,jailbreak_query,image_path\n"
                "unsafe,bypass,llm_transfer_attack/missing.png\n",
                encoding="utf-8",
            )

            with mock.patch.object(download_datasets, "DATA_DIR", data_dir):
                with self.assertRaises(SystemExit) as raised:
                    download_datasets.verify(False)

            self.assertIn(
                "data/JailBreakV_28K/llm_transfer_attack/missing.png",
                str(raised.exception),
            )

    def test_downloader_verification_reports_missing_mm_vet_reference(self) -> None:
        with TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            mm_vet_dir = data_dir / "MM-Vet"
            mm_vet_dir.mkdir(parents=True)
            (mm_vet_dir / "v1_0.png").write_bytes(b"marker")
            (mm_vet_dir / "mm-vet_metadata.json").write_text(
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

            with mock.patch.object(download_datasets, "DATA_DIR", data_dir):
                with self.assertRaises(SystemExit) as raised:
                    download_datasets.verify(False)

            self.assertIn("data/MM-Vet/v1_missing.png", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
