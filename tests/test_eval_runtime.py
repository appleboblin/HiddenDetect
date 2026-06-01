import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory


REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = REPO_ROOT / "code"
sys.path.insert(0, str(CODE_DIR))

from eval_runtime import finish_evaluation, validate_model_path


class EvalRuntimeTests(unittest.TestCase):
    def test_missing_local_model_path_exits_with_useful_message(self) -> None:
        with TemporaryDirectory() as tmpdir:
            with self.assertRaises(SystemExit) as raised:
                validate_model_path("model/missing-model", repo_root=Path(tmpdir))

            self.assertIn("Model path does not exist", str(raised.exception))
            self.assertIn("model/missing-model", str(raised.exception))

    def test_existing_model_directory_without_config_exits(self) -> None:
        with TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model" / "empty-model"
            model_dir.mkdir(parents=True)

            with self.assertRaises(SystemExit) as raised:
                validate_model_path("model/empty-model", repo_root=Path(tmpdir))

            self.assertIn("missing config.json", str(raised.exception))
            self.assertIn(str(model_dir / "config.json"), str(raised.exception))

    def test_existing_model_directory_with_config_passes(self) -> None:
        with TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model" / "valid-model"
            model_dir.mkdir(parents=True)
            (model_dir / "config.json").write_text("{}", encoding="utf-8")

            resolved = validate_model_path("model/valid-model", repo_root=Path(tmpdir))

            self.assertEqual(resolved, str(model_dir.resolve()))

    def test_empty_results_exit_without_writing_header_only_csv(self) -> None:
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results" / "llava-result.csv"

            with self.assertRaises(SystemExit) as raised:
                finish_evaluation(output_path, {}, ["XSTest", "FigTxt"])

            self.assertIn("No datasets completed successfully", str(raised.exception))
            self.assertFalse(output_path.exists())

    def test_partial_results_write_successful_rows_and_report_failures(self) -> None:
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results" / "llava-result.csv"
            stdout = StringIO()

            with redirect_stdout(stdout):
                finish_evaluation(output_path, {"XSTest": (0.81234, 0.92345)}, ["FigTxt"])

            self.assertEqual(
                output_path.read_text(encoding="utf-8").splitlines(),
                ["Dataset Name,AUPRC,AUROC", "XSTest,0.8123,0.9234"],
            )
            self.assertIn("Failed datasets (1): FigTxt", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
