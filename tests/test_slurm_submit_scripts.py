import os
import stat
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


class SlurmSubmitScriptsTest(unittest.TestCase):
    def test_logreg_wide_sweep_submits_matching_array_and_values(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script_path = repo_root / "scripts/slurm/submit_llava_logreg_c_wide_sweep.sh"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            capture_path = tmp_path / "sbatch-args.txt"
            mock_sbatch = tmp_path / "sbatch"
            mock_sbatch.write_text(
                textwrap.dedent(
                    """\
                    #!/bin/sh
                    printf '%s\\n' "$@" > "$SBATCH_CAPTURE"
                    printf '12345\\n'
                    """
                ),
                encoding="utf-8",
            )
            mock_sbatch.chmod(mock_sbatch.stat().st_mode | stat.S_IXUSR)

            env = os.environ.copy()
            env["PATH"] = f"{tmp_path}{os.pathsep}{env['PATH']}"
            env["SBATCH_CAPTURE"] = str(capture_path)

            result = subprocess.run(
                ["bash", str(script_path)],
                cwd=repo_root,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            args = capture_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--parsable", args)
        self.assertIn("--array=0-8", args)
        self.assertIn(
            "--export=ALL,LOGREG_C_VALUES=0.03125 0.0625 0.125 0.25 0.5 1 2 4 8",
            args,
        )
        self.assertEqual(args[-1], "scripts/slurm/run_llava_logreg_c_sweep.sbatch")
        self.assertIn("Submitted LogReg C wide sweep: job 12345", result.stdout)
        self.assertIn("Array range: 0-8", result.stdout)


if __name__ == "__main__":
    unittest.main()
