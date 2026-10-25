"""Record wrapper initialization and terminal states without ML dependencies."""
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def initialize(run, config):
    configuration = json.loads(config.read_text())
    save(run / "manifest.json", {"status": "running", "started_utc": now(),
         "pid": os.getppid(), "runner_pid": int(os.environ.get("CACHEGEN_SUPERVISOR_PID", os.getppid())),
         "configuration": configuration, "stages": {"preflight": "running"},
         "progress": {"stage": "preflight"}})
    state = os.environ.get("CACHEGEN_RUN_STATE_FILE")
    if state:
        save(Path(state), {"run_directory": str(run.resolve())})


def finish(run, code):
    path = run / "manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {"started_utc": now()}
    if code == 0 and manifest.get("status") != "completed":
        code = 1
        manifest["error"] = "Measurement exited without a completed manifest"
    if code:
        state = "interrupted" if code in (130, 143) else "failed"
        manifest.update(status=state, finished_utc=now(), progress={"stage": state})
        manifest.setdefault("error", f"Runner exited with code {code}")
        save(path, manifest)
    (run / "exit_code").write_text(f"{code}\n")
    (run / "process_finished_utc").write_text(now() + "\n")
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["initialize", "finish"])
    parser.add_argument("run", type=Path)
    parser.add_argument("value")
    args = parser.parse_args()
    if args.action == "initialize":
        initialize(args.run, Path(args.value))
        return 0
    return finish(args.run, int(args.value))


if __name__ == "__main__":
    raise SystemExit(main())
