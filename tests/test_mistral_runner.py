"""Test the real wrappers with a lightweight stand-in for the ML interpreter."""
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


class MistralRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="mistral checkout ")
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        (self.repo / "experiments/mistral").mkdir(parents=True)
        (self.repo / "scripts").mkdir()
        (self.repo / "mini_repro").mkdir()
        (self.repo / "env.yaml").write_text("name: test\n")
        for folder, patterns in [("experiments", ["*.py", "*.json"]),
                                 ("experiments/mistral", ["*.py", "*.sh", "*.json", "README.md"]),
                                 ("scripts", ["experiment-env.sh"])]:
            for pattern in patterns:
                for path in (REPO / folder).glob(pattern):
                    shutil.copy2(path, self.repo / folder / path.name)
        (self.repo / "mistral_repro").symlink_to("experiments/mistral", target_is_directory=True)
        interpreter = self.repo / "fake-python"
        interpreter.write_text(f"#!{sys.executable}\n" +
            "import os,pathlib,sys,time,signal\n"
            "if '-m' in sys.argv: raise SystemExit(int(os.environ.get('PREFLIGHT_EXIT','0')))\n"
            "if os.environ.get('IGNORE_TERM'): signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
            "pathlib.Path(os.environ['WORKER_PID']).write_text(str(os.getpid()))\n"
            "time.sleep(60)\n")
        interpreter.chmod(0o755)
        self.env = os.environ.copy()
        for key in list(self.env):
            if key.startswith("CACHEGEN_"):
                del self.env[key]
        self.env.update(CACHEGEN_PYTHON=str(interpreter), PYTHONPATH=str(self.repo),
                        PYTHONDONTWRITEBYTECODE="1", WORKER_PID=str(self.repo / "worker.pid"))

    def test_failed_preflight_has_manifest_and_is_visible_in_status(self):
        result = subprocess.run(["bash", str(self.repo / "mistral_repro/run.sh")],
                                cwd="/tmp", env={**self.env, "PREFLIGHT_EXIT": "7"}, capture_output=True, text=True)
        self.assertEqual(result.returncode, 7, result.stdout + result.stderr)
        status = subprocess.run([sys.executable, "-m", "experiments", "status", "mistral"],
                                cwd=self.repo, env=self.env, capture_output=True, text=True)
        data = json.loads(status.stdout)
        self.assertEqual(data["status"], "failed")
        self.assertEqual(data["exit_code"], 7)
        run = Path(data["run_directory"])
        self.assertTrue((run / "source/config.json").is_file())

    def test_direct_runner_termination_stops_worker_and_marks_interrupted(self):
        self.check_termination(["bash", str(self.repo / "mistral_repro/run.sh")])

    def test_cli_termination_stops_worker_and_marks_interrupted(self):
        self.check_termination([sys.executable, "-m", "experiments", "run", "mistral"])

    def test_hard_stop_cleans_stubborn_worker_and_records_terminal_state(self):
        self.env["IGNORE_TERM"] = "1"
        self.check_termination([sys.executable, "-m", "experiments", "run", "mistral"])

    def check_termination(self, command):
        process = subprocess.Popen(command, cwd=self.repo, env=self.env,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        worker = None
        try:
            marker = self.repo / "worker.pid"
            deadline = time.monotonic() + 5
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(.02)
            self.assertTrue(marker.exists(), "worker did not start")
            worker = int(marker.read_text())
            process.terminate()
            self.assertEqual(process.wait(timeout=15), 143)
            state = Path(f"/proc/{worker}/stat")
            deadline = time.monotonic() + 1
            while state.exists() and state.read_text().split()[2] != "Z" and time.monotonic() < deadline:
                time.sleep(.02)
            self.assertTrue(not state.exists() or state.read_text().split()[2] == "Z", "worker remains alive")
            run = (self.repo / "results/mistral/latest").resolve()
            self.assertEqual((run / "exit_code").read_text().strip(), "143")
            self.assertEqual(json.loads((run / "manifest.json").read_text())["status"], "interrupted")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            if worker is not None:
                try:
                    os.kill(worker, signal.SIGKILL)
                except ProcessLookupError:
                    pass
