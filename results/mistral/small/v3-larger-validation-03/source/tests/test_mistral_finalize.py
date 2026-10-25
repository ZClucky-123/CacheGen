"""Completed archives and source snapshots survive repeated postprocessing."""
import hashlib
import json
import shutil
import tempfile
import subprocess
import sys
import os
import unittest
from pathlib import Path

from experiments.mistral import finalize


class FinalizeTests(unittest.TestCase):
    def test_completed_archive_is_idempotent_and_preserves_source_snapshots(self):
        original = Path(__file__).resolve().parents[1] / "results/mistral/20261024T064543Z"
        if not original.exists():
            self.skipTest("historical fixture is not present in this checkout")
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            shutil.copytree(original, run)
            before = {str(p.relative_to(run)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in run.rglob("*") if p.is_file()}
            self.assertEqual(finalize.main([str(run)]), 0)
            after = {str(p.relative_to(run)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in run.rglob("*") if p.is_file()}
            self.assertEqual(before, after)

    def test_historical_snapshot_can_verify_changed_active_source(self):
        self.assertTrue(callable(getattr(finalize, "verify_source_fingerprints", None)),
                        "historical source validation is missing")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            (run / "source").mkdir(parents=True)
            active = root / "experiment.py"
            active.write_text("new source")
            (run / "source/experiment.py").write_text("original source")
            expected = hashlib.sha256(b"original source").hexdigest()
            (run / "metadata.json").write_text(json.dumps({"source_and_data_sha256": {str(active): expected}}))
            finalize.verify_source_fingerprints(run, root)
            (run / "source/experiment.py").write_text("corrupted snapshot")
            with self.assertRaises(RuntimeError):
                finalize.verify_source_fingerprints(run, root)

    def test_repository_relative_fingerprints_survive_checkout_move(self):
        self.assertTrue(callable(getattr(finalize, "verify_source_fingerprints", None)))
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            (repo / "src").mkdir()
            (repo / "src/tool.py").write_text("source")
            expected = hashlib.sha256(b"source").hexdigest()
            run = repo / "run"
            run.mkdir()
            (run / "metadata.json").write_text(json.dumps({
                "source_and_data_sha256": {"/previous-checkout/src/tool.py": expected},
                "source_and_data_relative_sha256": {"src/tool.py": expected}}))
            finalize.verify_source_fingerprints(run, repo)

    def test_failed_integrity_check_preserves_original_checksum_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "report.md").write_text("corrupted")
            original = json.dumps({"status": "completed", "files": {"report.md": {
                "bytes": 8, "sha256": hashlib.sha256(b"original").hexdigest()}}})
            (run / "artifacts.json").write_text(original)
            env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
            result = subprocess.run([sys.executable, str(Path(finalize.__file__)), str(run)],
                                    env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual((run / "artifacts.json").read_text(), original)


if __name__ == "__main__":
    unittest.main()
