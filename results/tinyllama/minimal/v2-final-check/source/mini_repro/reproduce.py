"""Run the registered TinyLlama experiment design in a fresh result directory."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = ROOT / "mini_repro/profiles/formal4070.json"
SECTIONS = {
    "quality": {"samples", "lengths", "continuation", "qa_per_position", "max_new_tokens"},
    "ttft": {"samples", "lengths", "bandwidth_gbps", "cachegen_levels", "warmup", "repeats"},
    "adaptive": {"cases", "length", "chunk_size", "traces", "bandwidth_scales", "slos",
                 "max_new_tokens", "warmup", "repeats"},
}


def load_profile(path):
    profile = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(profile) != {"version", "purpose", "seed", "quality_tolerances", *SECTIONS} or type(profile["version"]) is not int or profile["version"] != 1:
        raise ValueError("expected version 1 profile with quality, ttft and adaptive sections")
    if not isinstance(profile["purpose"], str) or not profile["purpose"]:
        raise ValueError("profile purpose must be nonempty")
    if type(profile["seed"]) is not int or profile["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    for section, expected in SECTIONS.items():
        values = profile[section]
        if not isinstance(values, dict) or set(values) != expected:
            raise ValueError(f"incorrect keys in {section}: expected {sorted(expected)}")
        for key, value in values.items():
            numbers = value if isinstance(value, list) else [value]
            if not numbers or any(type(n) not in (int, float) or not math.isfinite(n) or n < 0 for n in numbers):
                raise ValueError(f"{section}.{key} must contain finite nonnegative numbers")
            if key not in ("warmup", "samples", "qa_per_position") and any(n <= 0 for n in numbers):
                raise ValueError(f"{section}.{key} must be positive")
            if key not in ("bandwidth_gbps", "bandwidth_scales", "slos") and any(type(n) is not int for n in numbers):
                raise ValueError(f"{section}.{key} must contain integers")
            if key in ("lengths", "bandwidth_gbps", "cachegen_levels", "bandwidth_scales", "slos"):
                if not isinstance(value, list) or len(set(value)) != len(value):
                    raise ValueError(f"{section}.{key} must be a list of distinct values")
            elif isinstance(value, list):
                raise ValueError(f"{section}.{key} must be a scalar")
    quality, ttft, adaptive = (profile[s] for s in SECTIONS)
    if quality["samples"] + quality["qa_per_position"] == 0 or quality["continuation"] < 2:
        raise ValueError("quality requires cases and at least two continuation tokens")
    if quality["qa_per_position"] > 10 or adaptive["cases"] > 10:
        raise ValueError("only ten distinct synthetic facts are available; do not expand by repeating them")
    if ttft["samples"] < 1 or not set(ttft["cachegen_levels"]).issubset({1, 2, 3}):
        raise ValueError("TTFT requires samples and CacheGen levels 1..3")
    if max(quality["lengths"]) + max(quality["continuation"], quality["max_new_tokens"] + 64) > 2048:
        raise ValueError("quality context plus suffix may exceed TinyLlama's 2048-token window")
    if max(ttft["lengths"]) + 64 > 2048 or adaptive["length"] + adaptive["max_new_tokens"] + 64 > 2048:
        raise ValueError("TTFT/adaptive context plus question may exceed model window")
    tolerances = profile["quality_tolerances"]
    if not isinstance(tolerances, dict) or set(tolerances) != {"max_nll_delta", "max_qa_drop"}:
        raise ValueError("expected max_nll_delta and max_qa_drop thresholds")
    if any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in tolerances.values()) or tolerances["max_qa_drop"] > 1:
        raise ValueError("invalid quality tolerances")
    return profile


def build_jobs(profile):
    jobs = []
    for kind in ("quality", "ttft"):
        jobs.append({"name": kind, "kind": kind, "result": f"{kind}.json",
                     "parameters": {**profile[kind], "seed": profile["seed"]}, "status": "pending"})
    adaptive = profile["adaptive"]
    for scale in adaptive["bandwidth_scales"]:
        for slo in adaptive["slos"]:
            name = f"adaptive-scale-{scale}-slo-{slo}"
            params = {k: v for k, v in adaptive.items() if k not in ("bandwidth_scales", "slos")}
            params.update(bandwidth_scale=scale, slo=slo, seed=profile["seed"])
            jobs.append({"name": name, "kind": "adaptive", "result": f"{name}.json",
                         "parameters": params, "status": "pending"})
    return jobs


def command_for(job, run_dir):
    module = {"quality": "quality_size", "ttft": "ttft", "adaptive": "adaptive"}[job["kind"]]
    command = [sys.executable, "-u", "-m", f"mini_repro.{module}", "--results", str(run_dir / job["result"])]
    for key, value in job["parameters"].items():
        command.append("--" + key.replace("_", "-"))
        command.extend(str(item) for item in (value if isinstance(value, list) else [value]))
    return command


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare_run(output_root, run_id):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id) or run_id in (".", ".."):
        raise ValueError("run ID must be a simple directory name")
    run_dir = Path(output_root).resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_metadata():
    # Kept out of import and dry-run paths so plans work without CUDA access.
    import torch
    from mini_repro.run import MODEL_ID, MODEL_REVISION

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; use the configured local launcher")
    model_home = Path(os.environ.get("HF_HOME", ROOT / ".venv/huggingface")) / "hub"
    snapshot = model_home / "models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots" / MODEL_REVISION
    if not (snapshot / "model.safetensors").is_file():
        raise RuntimeError(f"pinned offline model snapshot missing: {snapshot}")
    source_paths = sorted((ROOT / "mini_repro").glob("*.py"))
    source_paths += sorted((ROOT / "LMCache/lmcache/storage_backend/serde").glob("cachegen*.py"))
    source_paths += sorted(path for path in (ROOT / "LMCache/third_party/torchac_cuda").rglob("*")
                           if path.suffix in (".cu", ".cuh", ".cpp", ".h") and "build" not in path.parts)
    source_paths += [ROOT / "test_data/longchat.jsonl", ROOT / "test_data/bw.jsonl"]
    package_names = ("torch", "transformers", "numpy", "torchac", "torchac_cuda", "lmcache")
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    git_status = subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).splitlines()
    import torchac_cuda
    extension_path = Path(torchac_cuda.__file__)
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in package_names},
        "gpu": {"name": torch.cuda.get_device_name(0), "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
                "capability": list(torch.cuda.get_device_capability(0)), "cuda_runtime": torch.version.cuda},
        "git": {"head": git_head, "status": git_status},
        "codec_binary": {"path": str(extension_path), "sha256": file_hash(extension_path)},
        "source_and_data_sha256": {str(path.relative_to(ROOT)): file_hash(path) for path in source_paths},
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION,
                  "files_sha256": {p.name: file_hash(p) for p in sorted(snapshot.iterdir()) if p.is_file()}},
        "limitations": ["TinyLlama exploratory quantization schedules", "current encoder has no anchor/delta path",
                        "TTFT local paced socket; receive then decode", "adaptive uses estimated network time"],
    }


def run_logged(command, log_path):
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            return process.wait()
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            process.stdout.close()


def validate_result(job, path):
    """Require every registered observation, even if a process exits with zero."""
    data = json.loads(path.read_text(encoding="utf-8"))
    params, kind = job["parameters"], job["kind"]
    if kind == "quality":
        expected_methods = {"raw", "uniform4", "uniform8", "cachegen1", "cachegen2", "cachegen3"}
        expected = {"continuation": params["samples"] * len(params["lengths"]),
                    "qa": params["qa_per_position"] * 3 * len(params["lengths"])}
        for field, count in expected.items():
            rows = data.get(field, [])
            if len(rows) != count or len({row["case_id"] for row in rows}) != count:
                raise ValueError(f"incomplete {field}: expected {count} unique cases")
            if any(Counter(item["method"] for item in row["methods"]) != Counter(expected_methods) for row in rows):
                raise ValueError(f"incomplete method pairs in {field}")
    else:
        count = params["samples"] * len(params["lengths"]) if kind == "ttft" else params["cases"]
        cases = data.get("cases", [])
        if len(cases) != count or len({case["case_id"] for case in cases}) != count:
            raise ValueError(f"expected {count} unique {kind} cases")
        if kind == "ttft":
            methods = ["text", "uniform8"] + [f"cachegen{level}" for level in params["cachegen_levels"]]
            expected = Counter({(method, bw): params["repeats"] for method in methods for bw in params["bandwidth_gbps"]})
            actual = lambda row: (row["method"], row["bandwidth_gbps"])
        else:
            strategies = ("adaptive", "adaptive_static", "fixed1", "fixed2", "fixed3", "text")
            expected = Counter({(strategy, trace): params["repeats"] for strategy in strategies for trace in range(params["traces"])})
            actual = lambda row: (row["strategy"], row["trace_id"])
        if any(Counter(actual(row) for row in case["measurements"]) != expected for case in cases):
            raise ValueError(f"incomplete repeated {kind} measurements")


def execute_jobs(run_dir, manifest, run_command=run_logged):
    for index, job in enumerate(manifest["jobs"], 1):
        job["command"] = command_for(job, run_dir)
        job["status"] = "running"
        write_json(run_dir / "manifest.json", manifest)
        print(f"[{index}/{len(manifest['jobs'])}] {job['name']}", flush=True)
        started = time.perf_counter()
        try:
            code = run_command(job["command"], run_dir / f"{job['name']}.log")
            if code == 0:
                validate_result(job, run_dir / job["result"])
        except KeyboardInterrupt:
            code = 130
            job["error"] = "execution interrupted"
        except Exception as error:
            code = 1
            job["error"] = str(error)
        job.update(exit_code=code, elapsed_seconds=round(time.perf_counter() - started, 3),
                   status="completed" if code == 0 else "interrupted" if code == 130 else "failed")
        if code:
            manifest["status"] = "interrupted" if code == 130 else "failed"
            write_json(run_dir / "manifest.json", manifest)
            return code
        write_json(run_dir / "manifest.json", manifest)
    manifest["status"] = "measurements_completed"
    write_json(run_dir / "manifest.json", manifest)
    return 0


def legacy_main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/reproduction")
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    parser.add_argument("--dry-run", action="store_true", help="print design/commands without CUDA or creating results")
    args = parser.parse_args(argv)
    try:
        profile = load_profile(args.profile)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    run_dir = args.output_root.resolve() / args.run_id
    jobs = build_jobs(profile)
    if args.dry_run:
        print(json.dumps({"profile": profile, "run_dir": str(run_dir),
                          "commands": [command_for(job, run_dir) for job in jobs]}, indent=2))
        return 0
    run_dir = prepare_run(args.output_root, args.run_id)
    manifest = {"profile": profile, "profile_path": str(args.profile.resolve()), "status": "initializing",
                "started_utc": datetime.now(timezone.utc).isoformat(), "jobs": jobs}
    write_json(run_dir / "manifest.json", manifest)
    try:
        write_json(run_dir / "metadata.json", collect_metadata())
        manifest["status"] = "running"
        code = execute_jobs(run_dir, manifest)
        if code:
            return code
        from mini_repro.report import generate_report
        generate_report(run_dir, **profile["quality_tolerances"])
        manifest["status"] = "completed"
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(run_dir / "manifest.json", manifest)
    except KeyboardInterrupt:
        manifest.update(status="interrupted", error="execution interrupted")
        write_json(run_dir / "manifest.json", manifest)
        return 130
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        write_json(run_dir / "manifest.json", manifest)
        raise
    print(f"Report: {run_dir / 'report.md'}", flush=True)
    return 0


def main(argv=None):
    """Current entrypoint; v1 is explicit and old profiles migrate to shared v2."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if '--legacy-v1' in argv:
        argv.remove('--legacy-v1')
        return legacy_main(argv)
    from experiments.suite import main as shared_main
    from experiments.config import load_config
    if '--profile' in argv:
        index = argv.index('--profile')
        profile = load_profile(argv[index + 1])
        del argv[index:index + 2]
        if profile['quality_tolerances'] != {'max_nll_delta': .02, 'max_qa_drop': .1}:
            raise ValueError('v2 fixes tolerances; use --legacy-v1 for a historical protocol')
        cfg = load_config(ROOT / 'experiments/configs/small.json', 'tinyllama')
        cfg['quality'].update(profile['quality'])
        cfg['ttft'].update({k: v for k, v in profile['ttft'].items() if k != 'cachegen_levels'})
        cfg['ttft']['methods'] = ['text', 'uniform8'] + [f'cachegen{n}' for n in profile['ttft']['cachegen_levels']]
        cfg['adaptive'].update(profile['adaptive'])
        cfg['seed'] = profile['seed']
        cfg['scale'] = 'minimal' if profile['purpose'] == 'pipeline-validation-only' else 'small'
        print('Migrating v1 profile to shared v2; NLL/QA/network protocol changes. Use --legacy-v1 for historical execution.', file=sys.stderr)
        return shared_main(argv, configuration=cfg)
    if '--resume' not in argv:
        argv = ['--model', 'tinyllama'] + argv
    return shared_main(argv)


if __name__ == '__main__':
    sys.exit(main())
