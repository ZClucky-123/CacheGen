"""One intentional TERM/resume of this exact readiness batch, at a 4K case."""
import hashlib
import json
import os
import signal
import sqlite3
import subprocess
import time
from pathlib import Path

ROOT = Path('/home/aaa/CacheGen')
RUN = ROOT / 'results/mistral/small/v3-larger-validation-03'
INITIAL = ROOT / '.local/launches/mistral-v3-readiness-03.json'
PROOF = ROOT / '.local/larger-readiness-recovery-before.json'

def read(path):
    return json.loads(path.read_text())

deadline = time.monotonic() + 3600
while time.monotonic() < deadline:
    if (RUN / 'selection.json').exists() and (RUN / 'records.sqlite3').exists():
        manifest = read(RUN / 'manifest.json')
        if manifest['status'] in ('failed', 'interrupted', 'completed'):
            raise RuntimeError(f'Unexpected state before intentional interruption: {manifest["status"]}')
        case = next(c for c in read(RUN / 'selection.json')['ttft'] if len(c['context_ids']) == 4096)
        prefix = f'ttft/{case["case_id"]}/'
        with sqlite3.connect(f'file:{RUN / "records.sqlite3"}?mode=ro', uri=True) as connection:
            count = connection.execute('SELECT COUNT(*) FROM observations WHERE key LIKE ?', (prefix + '%',)).fetchone()[0]
        if 3 <= count < 21:
            pid = manifest['runner_pid']
            cmdline = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
            if 'experiments.suite' not in cmdline or 'v3-larger-validation-03' not in cmdline:
                raise RuntimeError(f'Refusing to signal unexpected process {pid}: {cmdline}')
            os.kill(pid, signal.SIGTERM)
            for _ in range(300):
                if read(INITIAL)['status'] == 'interrupted':
                    break
                time.sleep(.2)
            else:
                raise RuntimeError('TERM did not cleanly finish in 60 seconds')
            after = read(RUN / 'manifest.json')
            if after['exit_code'] != 143:
                raise RuntimeError(f'Unexpected interrupt exit: {after}')
            with sqlite3.connect(f'file:{RUN / "records.sqlite3"}?mode=ro', uri=True) as connection:
                rows = dict(connection.execute('SELECT key, sha256 FROM observations'))
            prepared = {}
            for path in (RUN / 'pending-prepared').rglob('prepared.json'):
                obj = read(path)
                for packet in obj['packets']:
                    payload = path.parent / packet['file']
                    if hashlib.sha256(payload.read_bytes()).hexdigest() != packet['sha256']:
                        raise RuntimeError(f'Prepared payload corrupt before resume: {payload}')
                prepared[str(path.relative_to(RUN))] = obj
            proof = {'status': 'interrupted_verified', 'case_id': case['case_id'], 'target_tokens': 4096,
                     'rows_before_resume': rows, 'prepared_before_resume': prepared,
                     'manifest_before_resume': after, 'initial_launch': read(INITIAL)}
            PROOF.write_text(json.dumps(proof, ensure_ascii=False, indent=2) + '\n')
            result = subprocess.run(['python3', '-m', 'experiments.background', 'start', '--name',
                'mistral-v3-readiness-03-resume', '--', '--resume', str(RUN)], cwd=ROOT, text=True, capture_output=True)
            print(result.stdout, flush=True)
            if result.returncode:
                raise RuntimeError(result.stderr)
            print(f'Interrupted/resumed at {case["case_id"]}; retained {len(rows)} observations and {len(prepared)} prepared representations', flush=True)
            break
    time.sleep(.2)
else:
    raise RuntimeError('Timed out waiting for partially measured 4K case')
