"""Exercise public commands without loading models or requiring a GPU."""
import json
import fcntl
import os
import shutil
import subprocess
import sys
import time
import signal
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="cachegen workspace ")
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name) / "relocated checkout"
        (self.repo / "experiments").mkdir(parents=True)
        (self.repo / "LMCache").mkdir()
        (self.repo / "mini_repro").mkdir()
        (self.repo / "env.yaml").write_text("name: test\n")
        (self.repo / "jobs").mkdir()
        (self.repo / "jobs/echo.py").write_text(
            "import json,os,sys\n"
            "print(json.dumps({'args':sys.argv[1:],'cwd':os.getcwd(),"
            "'results':os.environ['CACHEGEN_RESULTS_DIR']}))\n"
        )
        self.registry = {
            "toy": {"description": "registered without dispatcher changes",
                    "command": ["{python}", "{repo}/jobs/echo.py"],
                    "results_subdir": "toy"}
        }
        self.save_registry()

    def save_registry(self):
        (self.repo / "experiments/registry.json").write_text(json.dumps(self.registry))

    def cli(self, *args, overrides=None, installed=None):
        env = os.environ.copy()
        for key in list(env):
            if key.startswith("CACHEGEN_"):
                del env[key]
        env.update(PYTHONPATH=str(installed or REPO), PYTHONDONTWRITEBYTECODE="1")
        if installed is None:
            env["CACHEGEN_REPO_ROOT"] = str(self.repo)
        env.update(overrides or {})
        return subprocess.run([sys.executable, "-m", "experiments", *args],
                              cwd=Path(self.temp.name), env=env,
                              capture_output=True, text=True)

    def test_new_registered_experiment_runs_with_literal_arguments(self):
        result = self.cli("run", "toy", "--", "a b", "$(do-not-execute)")
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data["args"], ["a b", "$(do-not-execute)"])
        self.assertEqual(data["cwd"], str(self.repo))
        self.assertEqual(data["results"], str(self.repo / "results"))

    def test_relocated_checkout_with_spaces_needs_no_absolute_root_setting(self):
        for source in (REPO / "experiments").glob("*"):
            if source.is_file() and source.suffix in {".py", ".json"}:
                shutil.copy2(source, self.repo / "experiments" / source.name)
        self.save_registry()
        result = self.cli("run", "toy", "--dry-run", installed=self.repo)
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data["cwd"], str(self.repo))
        self.assertEqual(data["command"][1], str(self.repo / "jobs/echo.py"))
        self.assertFalse((self.repo / "results").exists())

    def test_overrides_are_resolved_relative_to_checkout_without_side_effects(self):
        result = self.cli("run", "toy", "--dry-run", overrides={
            "CACHEGEN_RESULTS_DIR": "my measurements",
            "CACHEGEN_REPORTS_DIR": "my reports", "CACHEGEN_LOCAL_DIR": "my cache"})
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data["environment"]["CACHEGEN_RESULTS_DIR"], str(self.repo / "my measurements"))
        self.assertEqual(data["environment"]["CACHEGEN_REPORTS_DIR"], str(self.repo / "my reports"))
        self.assertEqual(data["environment"]["CACHEGEN_LOCAL_DIR"], str(self.repo / "my cache"))
        self.assertFalse((self.repo / "my cache").exists())

    def test_fresh_checkout_status_explains_no_run(self):
        result = self.cli("status", "toy")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "not_started")
        self.assertFalse((self.repo / "results").exists())

    def test_status_uses_configuration_counts_and_excludes_warmups(self):
        run = self.repo / "alternate/toy/batch"
        run.mkdir(parents=True)
        (run / "manifest.json").write_text(json.dumps({
            "status": "running", "started_utc": "2026-10-24T00:00:00+00:00",
            "progress": {"stage": "ttft"}, "configuration": {
                "quality": {"samples": 2, "lengths": [128]}, "qa": {"samples": 1},
                "ttft": {"samples": 1, "lengths": [128], "bandwidth_gbps": [1], "repeats": 2}}}))
        (run / "quality.json").write_text('{"continuation":[{},{}],"qa":[{}]}')
        (run / "ttft.json").write_text(json.dumps({"cases": [
            {"measurements": [{"phase": "warmup"}, {"phase": "measure"}]}]}))
        result = self.cli("status", "toy", overrides={"CACHEGEN_RESULTS_DIR": "alternate"})
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data["natural_contexts"], "2/2")
        self.assertEqual(data["qa_questions"], "1/1")
        self.assertEqual(data["ttft_measurements"], "1/8")

    def test_unknown_family_and_escaping_result_directory_are_rejected(self):
        result = self.cli("run", "unknown", "--dry-run")
        self.assertEqual(result.returncode, 2)
        self.registry["toy"]["results_subdir"] = "../outside"
        self.save_registry()
        result = self.cli("list")
        self.assertEqual(result.returncode, 2)
        self.assertFalse((self.repo / "results").exists())

    def test_resumed_run_has_live_elapsed_time_and_labels_previous_exit(self):
        run = self.repo / 'results/toy/resumed'
        run.mkdir(parents=True)
        (run / 'manifest.json').write_text(json.dumps({'status': 'running', 'started_utc': '2020-01-01T00:00:00+00:00',
            'finished_utc': '2020-01-01T00:01:00+00:00', 'protocol': 'cachegen-shared-v2', 'jobs': []}))
        (run / 'exit_code').write_text('143\n')
        result = self.cli('status', 'toy')
        data = json.loads(result.stdout)
        self.assertGreater(data['elapsed_minutes'], 60)
        self.assertEqual(data['previous_attempt_exit_code'], 143)
        self.assertNotIn('exit_code', data)

    def test_nested_active_batch_is_found_before_legacy_latest_link(self):
        legacy = self.repo / 'results/toy/old'
        current = self.repo / 'results/toy/larger/current'
        legacy.mkdir(parents=True)
        current.mkdir(parents=True)
        (legacy / 'manifest.json').write_text(json.dumps({'status': 'completed'}))
        (legacy.parent / 'latest').symlink_to('old')
        (current / 'manifest.json').write_text(json.dumps({'status': 'running', 'protocol': 'cachegen-shared-v3',
            'jobs': [], 'progress': {'completed': 9, 'expected': 100}}))
        result = self.cli('status', 'toy')
        data = json.loads(result.stdout)
        self.assertEqual(data['run_directory'], str(current.resolve()))
        self.assertEqual(data['progress']['completed'], 9)

    def test_child_failure_exit_code_is_preserved(self):
        self.registry["toy"]["command"] = ["{python}", "-c", "raise SystemExit(7)"]
        self.save_registry()
        result = self.cli("run", "toy")
        self.assertEqual(result.returncode, 7, result.stderr)

    def shell(self, command, overrides=None):
        env = os.environ.copy()
        for key in list(env):
            if key.startswith("CACHEGEN_"):
                del env[key]
        env.update(CACHEGEN_REPO_ROOT=str(self.repo), PYTHONDONTWRITEBYTECODE="1")
        env.update(overrides or {})
        return subprocess.run(["bash", "-ec", 'source "$1"; ' + command,
                               "bash", str(REPO / "scripts/experiment-env.sh")],
                              cwd=Path(self.temp.name), env=env, capture_output=True, text=True)

    def test_shared_shell_environment_preserves_paths_with_spaces_and_overrides(self):
        result = self.shell('printf "%s\\n" "$CACHEGEN_RESULTS_DIR" "$CACHEGEN_LOCAL_DIR" "$CACHEGEN_PYTHON"',
                            {"CACHEGEN_RESULTS_DIR": "new results", "CACHEGEN_LOCAL_DIR": "new cache",
                             "CACHEGEN_PYTHON": sys.executable})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(),
                         [str(self.repo / "new results"), str(self.repo / "new cache"), sys.executable])
        self.assertFalse((self.repo / "new cache").exists())

    def test_gpu_lock_rejects_second_measurement_process(self):
        local = self.repo / "my cache"
        local.mkdir()
        with (local / "gpu.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = self.shell("cachegen_lock_gpu", {"CACHEGEN_LOCAL_DIR": str(local)})
        self.assertEqual(result.returncode, 75, result.stderr)
        self.assertIn("GPU", result.stderr)
        self.assertFalse((self.repo / "results").exists())

    def test_terminating_dispatcher_stops_worker_and_preserves_nonzero_exit(self):
        marker = self.repo / "worker.pid"
        (self.repo / "jobs/echo.py").write_text(
            "import os,time,pathlib\n"
            "pathlib.Path('worker.pid').write_text(str(os.getpid()))\n"
            "time.sleep(60)\n")
        env = os.environ.copy()
        env.update(PYTHONPATH=str(REPO), CACHEGEN_REPO_ROOT=str(self.repo), PYTHONDONTWRITEBYTECODE="1")
        process = subprocess.Popen([sys.executable, "-m", "experiments", "run", "toy"],
                                   cwd=self.repo, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        worker = None
        try:
            deadline = time.monotonic() + 5
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(.02)
            self.assertTrue(marker.exists(), "worker did not start")
            worker = int(marker.read_text())
            process.send_signal(signal.SIGTERM)
            self.assertEqual(process.wait(timeout=5), 143)
            with self.assertRaises(ProcessLookupError):
                os.kill(worker, 0)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            if worker is not None:
                try:
                    os.kill(worker, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def test_termination_cleans_descendant_that_ignores_term_after_leader_exits(self):
        marker = self.repo / "descendant.pid"
        child_code = ("import os,pathlib,signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);"
                      "pathlib.Path('descendant.pid').write_text(str(os.getpid()));time.sleep(60)")
        (self.repo / "jobs/echo.py").write_text(
            "import subprocess,sys,time\nsubprocess.Popen([sys.executable,'-c'," + repr(child_code) + "])\ntime.sleep(60)\n")
        env = {**os.environ, "PYTHONPATH": str(REPO), "CACHEGEN_REPO_ROOT": str(self.repo),
               "PYTHONDONTWRITEBYTECODE": "1"}
        process = subprocess.Popen([sys.executable, "-m", "experiments", "run", "toy"],
                                   cwd=self.repo, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        descendant = None
        try:
            deadline = time.monotonic() + 5
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(.02)
            self.assertTrue(marker.exists())
            descendant = int(marker.read_text())
            process.terminate()
            self.assertEqual(process.wait(timeout=5), 143)
            state = Path(f"/proc/{descendant}/stat")
            deadline = time.monotonic() + 1
            while state.exists() and state.read_text().split()[2] != "Z" and time.monotonic() < deadline:
                time.sleep(.02)
            self.assertTrue(not state.exists() or state.read_text().split()[2] == "Z", "descendant remains alive")
        finally:
            if process.poll() is None:
                process.kill(); process.wait()
            if descendant is not None:
                try:
                    os.kill(descendant, signal.SIGKILL)
                except ProcessLookupError:
                    pass


if __name__ == "__main__":
    unittest.main()
