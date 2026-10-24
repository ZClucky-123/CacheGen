"""Shared paths and read-only status; this module never imports torch."""
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path


@dataclass(frozen=True)
class Workspace:
    repo: Path
    results: Path
    reports: Path
    local: Path

    @classmethod
    def discover(cls):
        override = os.environ.get("CACHEGEN_REPO_ROOT")
        repo = Path(override).expanduser().resolve() if override else Path(__file__).resolve().parents[1]
        if not (repo / "env.yaml").is_file() or not (repo / "mini_repro").is_dir():
            raise ValueError(f"Not a CacheGen checkout: {repo}")

        def location(name, default):
            path = Path(os.environ.get(name, default)).expanduser()
            return (repo / path).resolve() if not path.is_absolute() else path.resolve()

        return cls(repo, location("CACHEGEN_RESULTS_DIR", "results"),
                   location("CACHEGEN_REPORTS_DIR", "reports"),
                   location("CACHEGEN_LOCAL_DIR", ".local"))

    def environment(self):
        return {"CACHEGEN_REPO_ROOT": str(self.repo), "CACHEGEN_RESULTS_DIR": str(self.results),
                "CACHEGEN_REPORTS_DIR": str(self.reports), "CACHEGEN_LOCAL_DIR": str(self.local)}

    def registry(self):
        registry = json.loads((self.repo / "experiments/registry.json").read_text())
        if not isinstance(registry, dict) or not registry:
            raise ValueError("Experiment registry must be a nonempty mapping")
        for name, entry in registry.items():
            if not re.fullmatch(r"[a-z][a-z0-9_-]*", name) or not isinstance(entry, dict):
                raise ValueError(f"Invalid experiment entry: {name}")
            command = entry.get("command")
            if not isinstance(command, list) or not command or any(not isinstance(x, str) or not x for x in command):
                raise ValueError(f"Invalid command for {name}")
            folder = Path(entry.get("results_subdir", name))
            if folder.is_absolute() or ".." in folder.parts:
                raise ValueError(f"Results directory must stay inside results root: {name}")
        return registry

    def command(self, entry, arguments):
        default_python = self.repo / ".venv/bin/python"
        python = os.environ.get("CACHEGEN_PYTHON") or (str(default_python) if default_python.is_file() else sys.executable)
        values = {"repo": str(self.repo), "python": python}
        return [item.format(**values) for item in entry["command"]] + list(arguments)

    def status(self, name, entry, run_directory=None):
        runs = self.results / entry.get("results_subdir", name)
        if run_directory is not None:
            run = Path(run_directory).expanduser().resolve()
        elif (runs / "latest/manifest.json").is_file():
            run = (runs / "latest").resolve()
        else:
            candidates = sorted(p.parent for p in runs.glob("*/manifest.json") if p.parent.name != "latest")
            if not candidates:
                return {"experiment": name, "status": "not_started", "results_directory": str(runs)}
            run = candidates[-1].resolve()
        manifest = json.loads((run / "manifest.json").read_text())
        result = {"experiment": name, "run_directory": str(run), "status": manifest["status"],
                  "progress": manifest.get("progress", manifest.get("stages", {}))}
        if manifest.get("started_utc"):
            started = datetime.fromisoformat(manifest["started_utc"])
            end = datetime.fromisoformat(manifest.get("finished_utc", datetime.now(timezone.utc).isoformat()))
            result.update(started_china_time=started.astimezone(timezone(timedelta(hours=8))).isoformat(),
                          elapsed_minutes=round((end-started).total_seconds()/60, 1))
        config = manifest.get("configuration", {})
        if (run / "quality.json").exists():
            quality = json.loads((run / "quality.json").read_text())
            spec = config.get("quality", {})
            expected = spec.get("samples", 0) * len(spec.get("lengths", []))
            result["natural_contexts"] = f"{len(quality.get('continuation', []))}/{expected}"
            result["qa_questions"] = f"{len(quality.get('qa', []))}/{config.get('qa', {}).get('samples', 0)}"
        if (run / "ttft.json").exists():
            timing = json.loads((run / "ttft.json").read_text())
            cases = timing.get("cases", []) + [timing.get("current_case", {})]
            count = sum(row.get("phase", "measure") == "measure"
                        for case in cases for row in case.get("measurements", []))
            spec = config.get("ttft", {})
            expected = (spec.get("samples", 0) * len(spec.get("lengths", [])) *
                        len(spec.get("bandwidth_gbps", [])) * len(spec.get("methods", [1, 2, 3, 4])) *
                        spec.get("repeats", 0))
            result["ttft_measurements"] = f"{count}/{expected}"
        if (run / "artifacts.json").exists():
            result["artifacts"] = json.loads((run / "artifacts.json").read_text())["status"]
        if (run / "exit_code").exists():
            result["exit_code"] = int((run / "exit_code").read_text())
        return result
