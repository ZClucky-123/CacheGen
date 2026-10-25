"""Run/resume the shared versioned protocol in an immutable, unique batch."""
import argparse
import fcntl
import json
import os
import random
import re
import signal
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from experiments.config import jobs, load_config, validate_config, TASK_KINDS
from experiments.records import Journal, expected_keys, save, fingerprint, validate_task_records
from experiments.workspace import Workspace


def now():
    return datetime.now(timezone.utc).isoformat()


class Tee:
    def __init__(self, console, stream):
        self.console, self.stream = console, stream

    def write(self, value):
        self.console.write(value)
        self.stream.write(value)
        self.flush()

    def flush(self):
        self.console.flush()
        self.stream.flush()


def run(args, workspace, cfg):
    from experiments.artifacts import (code_data_identity, runtime_identity, preflight,
                                        collect_metadata, sha256, seal, verify)
    import torch
    from experiments.engine import Engine
    from experiments.datasets import selection
    from experiments import tasks
    workspace.local.mkdir(parents=True, exist_ok=True)
    lock = (workspace.local / 'gpu.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        print('Another CacheGen GPU measurement is running.', file=sys.stderr)
        return 75
    run_dir = args.resume.resolve() if args.resume else (args.output_root or workspace.results / cfg['model']['name'] / cfg['scale']) / args.run_id
    journal = None
    previous_handlers = {}
    output = None
    previous_stdout, previous_stderr = sys.stdout, sys.stderr
    # A resume mismatch must not mutate a prior batch's manifest or exit code.
    initialized = False
    manifest = None
    try:
        torch.set_num_threads(cfg.get('cpu_threads', 8))
        torch.manual_seed(cfg['seed'])
        random.seed(cfg['seed'])
        current = code_data_identity(workspace.repo, cfg)
        if args.resume:
            manifest = json.loads((run_dir / 'manifest.json').read_text())
            if manifest['identity']['inputs'] != current:
                raise ValueError('Resume rejected: config/code/data fingerprints changed; use a new run ID')
            if manifest['status'] == 'completed':
                print(f'Already completed; verified {verify(run_dir)} artifacts: {run_dir}')
                return 0
        else:
            run_dir.mkdir(parents=True, exist_ok=False)
            manifest = {'protocol': cfg['protocol'], 'status': 'initializing', 'started_utc': now(),
                'configuration': cfg, 'jobs': jobs(cfg), 'identity': {'inputs': current}, 'attempts': []}
            save(run_dir / 'manifest.json', manifest)
            save(run_dir / 'config.json', cfg)
        initialized = True
        output = (run_dir / 'run.log').open('a', encoding='utf-8', buffering=1)
        sys.stdout, sys.stderr = Tee(previous_stdout, output), Tee(previous_stderr, output)
        print(json.dumps({'event': 'start', 'time_utc': now(), 'run_directory': str(run_dir), 'resume': bool(args.resume)}), flush=True)
        for sig in (signal.SIGTERM, signal.SIGINT):
            def interrupted(signum, _frame):
                raise KeyboardInterrupt(signum)
            previous_handlers[sig] = signal.signal(sig, interrupted)
        manifest.update(status='running', pid=os.getpid(), runner_pid=os.getpid(), progress={'stage': 'preflight'})
        manifest['attempts'].append({'started_utc': now(), 'pid': os.getpid(), 'resume': bool(args.resume)})
        manifest.pop('error', None)
        manifest.pop('finished_utc', None)
        manifest.pop('exit_code', None)
        save(run_dir / 'manifest.json', manifest)
        state_file = os.environ.get('CACHEGEN_RUN_STATE_FILE')
        if state_file:
            save(Path(state_file), {'run_directory': str(run_dir)})
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required for real measurements')
        engine = Engine(cfg['model'], load=False)
        selected = selection(workspace.repo, cfg, engine.tokenizer)
        preflight(engine, cfg, selected, run_dir)
        attempt_dir = run_dir / 'attempts' / f'{len(manifest["attempts"]):03d}'
        attempt_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(run_dir / 'preflight.json', attempt_dir / 'preflight.json')
        expected = expected_keys(cfg, selected, manifest['jobs'])
        if args.resume:
            meta = json.loads((run_dir / 'metadata.json').read_text())
            model_hashes = {p.name: sha256(p) for p in engine.snapshot.iterdir() if p.is_file()}
            if model_hashes != meta['model_files_sha256'] or runtime_identity() != meta['runtime']:
                raise ValueError('Resume rejected: model or execution environment changed')
            if fingerprint(selected) != manifest['identity']['selection']:
                raise ValueError('Resume rejected: data selection/tokenizer changed')
        else:
            save(run_dir / 'selection.json', selected)
            manifest['identity'].update(selection=fingerprint(selected), runtime=runtime_identity())
            meta = collect_metadata(workspace.repo, run_dir, engine, cfg, manifest['identity'])
        save(run_dir / 'expected.json', expected)
        journal = Journal(run_dir / 'records.sqlite3', manifest['identity'], attempt=len(manifest['attempts']))
        manifest['expected_observations'] = sum(map(len, expected.values()))
        manifest['progress'] = {'stage': 'calibration', 'completed': len(journal.keys), 'expected': manifest['expected_observations']}
        save(run_dir / 'manifest.json', manifest)
        parent = run_dir.parent
        latest = parent / 'latest'
        if latest.is_symlink() or not latest.exists():
            latest.unlink(missing_ok=True)
            latest.symlink_to(run_dir.name)
        if not (run_dir / 'calibration.json').is_file() or args.resume:
            tasks.calibrate(cfg, run_dir)
        shutil.copy2(run_dir / 'calibration.json', attempt_dir / 'calibration.json')
        manifest['progress']['stage'] = 'load_model'
        save(run_dir / 'manifest.json', manifest)
        engine.load()
        meta.update(attention_implementation=engine.model.config._attn_implementation,
                    attention_class=type(engine.model.model.layers[0].self_attn).__name__)
        save(run_dir / 'metadata.json', meta)
        torch.cuda.reset_peak_memory_stats()
        if not (run_dir / 'cache_checks.json').is_file() or json.loads((run_dir / 'cache_checks.json').read_text())['status'] != 'passed':
            manifest['progress']['stage'] = 'cache_checks'
            save(run_dir / 'manifest.json', manifest)
            with torch.inference_mode():
                tasks.cache_checks(engine, cfg, selected, run_dir)
        def progress(task, key):
            manifest['progress'] = {'stage': task, 'current_observation': key,
                                    'completed': len(journal.keys), 'expected': manifest['expected_observations']}
            save(run_dir / 'manifest.json', manifest)
        with torch.inference_mode():
            for job in manifest['jobs']:
                if set(expected[job['name']]).issubset(journal.keys):
                    validate_task_records(journal, job, expected[job['name']])
                    job['status'] = 'completed'
                    continue
                job.update(status='running', started_utc=now())
                job.pop('exit_code', None)
                save(run_dir / 'manifest.json', manifest)
                if job['kind'] not in TASK_KINDS:
                    raise ValueError(f'Unknown task kind: {job["kind"]}')
                getattr(tasks, job['kind'])(engine, cfg, selected, run_dir, journal, job, progress)
                count = validate_task_records(journal, job, expected[job['name']])
                job.update(status='completed', observations=count, finished_utc=now(), exit_code=0)
                save(run_dir / 'manifest.json', manifest)
                print(json.dumps({'event': 'task_completed', 'task': job['name'], 'observations': count, 'time_utc': now()}), flush=True)
        journal.validate([key for keys in expected.values() for key in keys])
        journal.export(run_dir / 'records.jsonl')
        if code_data_identity(workspace.repo, cfg) != current:
            raise ValueError('Code/data changed during measurement; batch cannot be marked completed')
        from experiments.analysis import report
        manifest['progress']['stage'] = 'report'
        save(run_dir / 'manifest.json', manifest)
        report(run_dir, cfg, journal, expected)
        journal.close()
        journal = None
        manifest.update(status='completed', finished_utc=now(), exit_code=0,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(), peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
            progress={'stage': 'completed', 'completed': manifest['expected_observations'], 'expected': manifest['expected_observations']})
        manifest['attempts'][-1].update(finished_utc=now(), exit_code=0)
        save(run_dir / 'manifest.json', manifest)
        (run_dir / 'exit_code').write_text('0\n')
        print(f'Measurements and report complete; finalizing archive: {run_dir}', flush=True)
        return 0
    except BaseException as error:
        if not initialized:
            raise
        code = (128 + (error.args[0] if error.args and type(error.args[0]) is int else 2)) if isinstance(error, KeyboardInterrupt) else 1
        manifest.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=repr(error),
                        finished_utc=now(), exit_code=code)
        for job in manifest['jobs']:
            if job.get('status') == 'running':
                job.update(status=manifest['status'], exit_code=code)
        if manifest['attempts']:
            manifest['attempts'][-1].update(finished_utc=now(), exit_code=code)
        save(run_dir / 'manifest.json', manifest)
        (run_dir / 'exit_code').write_text(f'{code}\n')
        with (run_dir / 'failures.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'time_utc': now(), 'error': repr(error), 'progress': manifest.get('progress'),
                                      'traceback': traceback.format_exc()}, ensure_ascii=False) + '\n')
        traceback.print_exc()
        return code
    finally:
        if journal is not None:
            journal.export(run_dir / 'partial-records.jsonl')
            journal.close()
        if initialized and output is not None:
            output.flush()
        sys.stdout, sys.stderr = previous_stdout, previous_stderr
        if output is not None:
            output.close()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        if initialized and manifest.get('status') == 'completed':
            try:
                seal(run_dir)
                print(f'COMPLETED {run_dir}; {verify(run_dir)} artifacts verified', flush=True)
            except Exception as error:
                manifest.update(status='failed', error=f'Archive finalization: {error}', exit_code=1)
                save(run_dir / 'manifest.json', manifest)
                (run_dir / 'exit_code').write_text('1\n')
                traceback.print_exc()
                lock.close()
                return 1
        lock.close()


def main(argv=None, configuration=None):
    workspace = Workspace.discover()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--model', choices=('tinyllama', 'mistral'))
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--run-id', default=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--preflight-only', action='store_true', help='check actual data/window/resources/plots without loading weights or measuring')
    parser.add_argument('--verify', type=Path)
    args = parser.parse_args(argv)
    if args.verify:
        from experiments.artifacts import verify
        print(json.dumps({'verified_files': verify(args.verify), 'run_directory': str(args.verify)}))
        return 0
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.run_id):
        parser.error('run ID must be a simple directory name')
    if args.resume and (args.config or args.model or args.output_root):
        parser.error('Resume uses the recorded configuration; do not override config/model/output-root')
    path = args.resume / 'config.json' if args.resume else args.config or workspace.repo / 'experiments/configs/small.json'
    cfg = validate_config(configuration, args.model) if configuration is not None else load_config(path, args.model)
    if args.dry_run:
        print(json.dumps({'configuration': cfg, 'jobs': jobs(cfg),
                          'run_directory': str(args.resume or (args.output_root or workspace.results / cfg['model']['name'] / cfg['scale']) / args.run_id),
                          'mode': 'plan_only_no_gpu_or_writes'}, ensure_ascii=False, indent=2))
        return 0
    if args.preflight_only:
        return check_configuration(workspace, cfg)
    return run(args, workspace, cfg)


def check_configuration(workspace, cfg):
    from experiments.artifacts import preflight, code_data_identity
    from experiments.engine import Engine
    from experiments.datasets import selection
    import torch
    workspace.local.mkdir(parents=True, exist_ok=True)
    with (workspace.local / 'gpu.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('Another CacheGen GPU measurement is running.', file=sys.stderr)
            return 75
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA resource inspection requires the target GPU')
        engine = Engine(cfg['model'], load=False)
        selected = selection(workspace.repo, cfg, engine.tokenizer)
        target = workspace.local / 'checked-configs' / fingerprint(cfg)[:16]
        target.mkdir(parents=True, exist_ok=True)
        resources = preflight(engine, cfg, selected, target)
        expected = expected_keys(cfg, selected, jobs(cfg))
        result = {'status': 'preflight_only_passed', 'time_utc': now(), 'model': cfg['model'],
            'configuration_sha256': fingerprint(cfg), 'counts': {k: len(v) for k, v in selected.items()},
            'source_and_data_fingerprint': fingerprint(code_data_identity(workspace.repo, cfg)),
            'triviaqa_source_ids': [r['source_id'] for r in selected['triviaqa']],
            'expected_observations': {k: len(v) for k, v in expected.items()},
            'resources': resources, 'model_loaded': False, 'measurements_executed': False}
        save(target / 'check.json', result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
