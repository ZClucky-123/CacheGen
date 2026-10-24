"""Detached tmux runs with durable launch status and no dependence on SSH/IDE."""
import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from experiments.records import save
from experiments.workspace import Workspace


def main(argv=None):
    w = Workspace.discover()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('start', 'status', 'worker'))
    p.add_argument('--name', required=True)
    a, remaining = p.parse_known_args(argv)
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]*', a.name):
        p.error('Use a simple tmux session name')
    folder = w.local / 'launches'
    state = folder / f'{a.name}.json'
    if a.action == 'status':
        data = json.loads(state.read_text())
        data['tmux_alive'] = subprocess.run(['tmux', '-L', 'cachegen', 'has-session', '-t', a.name], capture_output=True).returncode == 0
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return 0
    if a.action == 'worker':
        data = json.loads(state.read_text())
        data.update(status='running', pid=os.getpid())
        save(state, data)
        with (folder / f'{a.name}.log').open('a', buffering=1) as log:
            result = subprocess.run(data['command'], cwd=w.repo, env={**os.environ, **data['environment']}, stdout=log, stderr=subprocess.STDOUT)
        data.update(status='completed' if result.returncode == 0 else 'interrupted' if result.returncode in (130, 143) else 'failed', exit_code=result.returncode,
                    finished_utc=datetime.now(timezone.utc).isoformat())
        save(state, data)
        run = Path(data['run_directory'])
        if run.is_dir():
            destination = run / 'launches'
            destination.mkdir(exist_ok=True)
            log_path = destination / f'{a.name}.log'
            state_path = destination / f'{a.name}.json'
            shutil.copy2(folder / f'{a.name}.log', log_path)
            save(state_path, data)
            if result.returncode == 0:
                from experiments.artifacts import verify, sha256
                verify(run)
                archived = json.loads((run / 'artifacts.json').read_text())
                for path in (log_path, state_path):
                    archived['files'][str(path.relative_to(run))] = {'bytes': path.stat().st_size, 'sha256': sha256(path)}
                save(run / 'artifacts.json', archived)
                verify(run)
        return result.returncode
    if not shutil.which('tmux'):
        p.error('tmux is required for detached launch')
    if state.exists():
        p.error('Launch name already exists; use a new name, including for resume attempts')
    args = remaining[1:] if remaining[:1] == ['--'] else remaining
    if not args:
        p.error('Pass suite arguments after --, e.g. -- --model mistral --config experiments/configs/small.json')
    # Validate without loading CUDA or creating result directories before detaching.
    command = ['bash', str(w.repo / 'experiments/run.sh'), *args]
    plan = json.loads(subprocess.check_output(command + ['--dry-run'], cwd=w.repo, text=True))
    folder.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k in ('CUDA_VISIBLE_DEVICES', 'HF_HOME', 'HF_HUB_CACHE',
           'CUDA_HOME', 'TORCH_EXTENSIONS_DIR', 'OMP_NUM_THREADS', 'CACHEGEN_PYTHON', 'CACHEGEN_PLOT_PYTHON')}
    env.update(w.environment())
    save(state, {'status': 'launching', 'command': command, 'environment': env, 'run_directory': plan['run_directory'],
                 'started_utc': datetime.now(timezone.utc).isoformat(), 'log': str(folder / f'{a.name}.log')})
    worker = shlex.join([sys.executable, '-m', 'experiments.background', 'worker', '--name', a.name])
    result = subprocess.run(['tmux', '-L', 'cachegen', '-f', '/dev/null', 'new-session', '-d', '-s', a.name,
                             '-c', str(w.repo), worker], capture_output=True, text=True)
    if result.returncode:
        save(state, {'status': 'launch_failed', 'error': result.stderr, 'exit_code': result.returncode})
        raise RuntimeError(result.stderr)
    print(json.dumps({'session': a.name, 'log': str(folder / f'{a.name}.log'),
        'attach': f'tmux -L cachegen attach -t {a.name}',
        'status': f'python3 -m experiments.background status --name {a.name}'}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
