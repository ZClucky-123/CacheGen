"""Run a child process group and forward interruption without orphaning workers."""
import argparse
import os
import signal
import subprocess
import time
import json
import sys
import tempfile
from pathlib import Path


def run_process(command, cwd=None, env=None):
    child = None
    received = []
    deadline = None

    def forward(signum, _frame):
        nonlocal deadline
        received.append(signum)
        deadline = deadline or time.monotonic() + 10
        if child is not None:
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass

    previous = {sig: signal.signal(sig, forward) for sig in (signal.SIGINT, signal.SIGTERM)}
    environment = dict(os.environ if env is None else env)
    local = environment.get("CACHEGEN_LOCAL_DIR")
    supervision = Path(local) / "supervision" if local else None
    if supervision is not None:
        supervision.mkdir(parents=True, exist_ok=True)
    state = tempfile.TemporaryDirectory(prefix="run-", dir=supervision)
    state_file = Path(state.name) / "state.json"
    try:
        environment["CACHEGEN_SUPERVISOR_PID"] = str(os.getpid())
        environment["CACHEGEN_RUN_STATE_FILE"] = str(state_file)
        child = subprocess.Popen(command, cwd=cwd, env=environment, start_new_session=True)
        for signum in received:
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass
        while True:
            try:
                code = child.wait(timeout=.2)
                if received:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                break
            except subprocess.TimeoutExpired:
                if deadline is not None and time.monotonic() >= deadline:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
        if received:
            return 128 + received[0]
        return code if code >= 0 else 128 - code
    finally:
        if received and state_file.is_file():
            try:
                from .lifecycle import finish
                run = Path(json.loads(state_file.read_text())["run_directory"])
                manifest = json.loads((run / "manifest.json").read_text())
                if manifest.get("status") == "running" or not (run / "exit_code").exists():
                    finish(run, 128 + received[0])
            except (OSError, ValueError, KeyError) as error:
                print(f"Could not record interrupted run: {error}", file=sys.stderr)
        state.cleanup()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    command = parser.parse_args().command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("a command is required")
    return run_process(command)


if __name__ == "__main__":
    raise SystemExit(main())
