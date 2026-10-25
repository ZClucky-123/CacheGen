"""Compatibility wrappers forward to the shared runner without shell interpolation."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


class MistralRunnerTests(unittest.TestCase):
    def test_wrappers_share_runner_and_preserve_arguments_and_exit_code(self):
        with tempfile.TemporaryDirectory(prefix="shared interpreter ") as directory:
            fake = Path(directory) / "python"
            fake.write_text("#!" + sys.executable + "\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\nraise SystemExit(7)\n")
            fake.chmod(0o755)
            env = {**os.environ, 'CACHEGEN_PYTHON': str(fake)}
            for entry in ('experiments/mistral/run.sh', 'mistral_repro/run.sh'):
                result = subprocess.run(['bash', str(REPO / entry), '--config', 'a b$(literal).json'],
                                        cwd='/tmp', env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 7, result.stderr)
                self.assertEqual(json.loads(result.stdout), ['-u', '-m', 'experiments.suite', '--model', 'mistral', '--config', 'a b$(literal).json'])

    def test_resume_does_not_override_recorded_model(self):
        with tempfile.TemporaryDirectory() as directory:
            fake = Path(directory) / 'python'
            fake.write_text("#!" + sys.executable + "\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n")
            fake.chmod(0o755)
            result = subprocess.run(['bash', str(REPO / 'experiments/mistral/run.sh'), '--resume', 'some run'],
                env={**os.environ, 'CACHEGEN_PYTHON': str(fake)}, capture_output=True, text=True)
            args = json.loads(result.stdout)
            self.assertNotIn('--model', args)
            self.assertEqual(args[-2:], ['--resume', 'some run'])
