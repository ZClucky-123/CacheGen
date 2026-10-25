import copy
import json
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mini_repro.reproduce import build_jobs, execute_jobs, load_profile, main, prepare_run


PROFILE = Path(__file__).resolve().parents[1] / "mini_repro/profiles/formal4070.json"


class ReproduceTests(unittest.TestCase):
    def test_formal_design_has_four_adaptive_conditions_and_repeated_timings(self):
        profile = load_profile(PROFILE)
        jobs = build_jobs(profile)
        self.assertEqual([j["kind"] for j in jobs], ["quality", "ttft"] + ["adaptive"] * 4)
        self.assertEqual(jobs[1]["parameters"]["repeats"], 10)
        self.assertEqual(jobs[1]["parameters"]["cachegen_levels"], [2, 3])
        self.assertEqual(len({j["result"] for j in jobs}), len(jobs))

    def test_rejects_nonfinite_bandwidth_and_duplicated_fact_expansion(self):
        profile = load_profile(PROFILE)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for section, key, value in (("adaptive", "bandwidth_scales", [float("nan")]),
                                        ("quality", "qa_per_position", 11)):
                changed = copy.deepcopy(profile)
                changed[section][key] = value
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    load_profile(path)

    def test_existing_run_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            run = prepare_run(Path(directory), "known")
            marker = run / "important.json"
            marker.write_text("old result")
            with self.assertRaises(FileExistsError):
                prepare_run(Path(directory), "known")
            self.assertEqual(marker.read_text(), "old result")

    def test_distinct_close_bandwidth_scales_cannot_overwrite_artifacts(self):
        profile = load_profile(PROFILE)
        profile["adaptive"]["bandwidth_scales"] = [0.01, 0.010000001]
        jobs = build_jobs(profile)
        self.assertEqual(len({j["result"] for j in jobs}), len(jobs))

    def test_dry_run_never_collects_gpu_metadata_or_creates_results(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "results"
            with patch("mini_repro.reproduce.collect_metadata", side_effect=AssertionError("GPU touched")), \
                    patch("sys.stdout", new=io.StringIO()):
                self.assertEqual(main(["--profile", str(PROFILE), "--output-root", str(target),
                                       "--dry-run"]), 0)
            self.assertFalse(target.exists())

    def test_failed_job_preserves_completed_result_and_marks_pending(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            profile = load_profile(PROFILE)
            profile["quality"].update(samples=1, lengths=[512], qa_per_position=0)
            jobs = build_jobs(profile)[:3]
            manifest = {"jobs": jobs, "status": "running"}
            calls = []

            def runner(command, log_path):
                calls.append(command)
                if len(calls) == 1:
                    data = {"finished": True, "qa": [], "continuation": [{"case_id": "a", "methods": [
                        {"method": method} for method in ("raw", "uniform4", "uniform8", "cachegen1", "cachegen2", "cachegen3")]}]}
                    Path(command[command.index("--results") + 1]).write_text(json.dumps(data))
                    return 0
                return 7

            self.assertEqual(execute_jobs(run, manifest, run_command=runner), 7)
            saved = json.loads((run / "manifest.json").read_text())
            self.assertEqual(saved["status"], "failed")
            self.assertEqual([j["status"] for j in saved["jobs"]], ["completed", "failed", "pending"])
            self.assertTrue(json.loads((run / jobs[0]["result"]).read_text())["finished"])

    def test_successful_exit_with_partial_results_is_a_failed_experiment(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            jobs = build_jobs(load_profile(PROFILE))[:1]
            manifest = {"jobs": jobs, "status": "running"}

            def runner(command, log_path):
                Path(command[command.index("--results") + 1]).write_text('{"continuation":[],"qa":[]}')
                return 0

            self.assertEqual(execute_jobs(run, manifest, run_command=runner), 1)
            self.assertEqual(manifest["jobs"][0]["status"], "failed")

    def test_interruption_is_recorded_and_does_not_start_remaining_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            manifest = {"jobs": build_jobs(load_profile(PROFILE)), "status": "running"}
            def runner(command, log_path):
                raise KeyboardInterrupt()

            self.assertEqual(execute_jobs(run, manifest, run_command=runner), 130)
            saved = json.loads((run / "manifest.json").read_text())
            self.assertEqual(saved["status"], "interrupted")
            self.assertEqual(saved["jobs"][0]["status"], "interrupted")
            self.assertEqual(saved["jobs"][1]["status"], "pending")


if __name__ == "__main__":
    unittest.main()
