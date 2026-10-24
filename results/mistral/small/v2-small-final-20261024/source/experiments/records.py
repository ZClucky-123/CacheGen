"""Durable per-observation checkpoints and strict matrix completeness."""
import hashlib
import json
import sqlite3
import math
from datetime import datetime, timezone
from pathlib import Path


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


class Journal:
    def __init__(self, path, identity, attempt=None):
        self.attempt = attempt
        self.connection = sqlite3.connect(path)
        self.connection.execute('PRAGMA journal_mode=WAL')
        self.connection.execute('PRAGMA synchronous=FULL')
        self.connection.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
        self.connection.execute('CREATE TABLE IF NOT EXISTS observations (key TEXT PRIMARY KEY, task TEXT NOT NULL, body TEXT NOT NULL, sha256 TEXT NOT NULL)')
        old = self.connection.execute('SELECT value FROM metadata WHERE key="identity"').fetchone()
        encoded = canonical(identity)
        if old and old[0] != encoded:
            self.connection.close()
            raise ValueError('Checkpoint identity mismatch: configuration/code/data/model/environment changed')
        self.connection.execute('INSERT OR IGNORE INTO metadata VALUES ("identity", ?)', (encoded,))
        self.connection.commit()
        self.keys = set()
        for key, body, digest in self.connection.execute('SELECT key, body, sha256 FROM observations'):
            if hashlib.sha256(body.encode()).hexdigest() != digest:
                self.connection.close()
                raise ValueError(f'Checkpoint observation corrupted: {key}')
            self.keys.add(key)

    def add(self, key, task, value):
        body = canonical({**value, 'observed_utc': datetime.now(timezone.utc).isoformat(), 'attempt': self.attempt})
        if key in self.keys:
            raise ValueError(f'Duplicate observation: {key}')
        self.connection.execute('INSERT INTO observations VALUES (?, ?, ?, ?)',
                                (key, task, body, hashlib.sha256(body.encode()).hexdigest()))
        self.connection.commit()
        self.keys.add(key)

    def rows(self, task=None):
        query = 'SELECT key, task, body FROM observations'
        cursor = self.connection.execute(query + (' WHERE task=?' if task else '') + ' ORDER BY rowid',
                                         (task,) if task else ())
        return [{'record_id': key, 'task': name, **json.loads(body)} for key, name, body in cursor]

    def validate(self, expected, task=None):
        actual = {row['record_id'] for row in self.rows(task)}
        if actual != set(expected):
            raise ValueError(f'Incomplete/unexpected observations: missing={sorted(set(expected)-actual)[:8]}, extra={sorted(actual-set(expected))[:8]}')
        return len(actual)

    def export(self, path):
        with Path(path).open('w', encoding='utf-8') as stream:
            for row in self.rows():
                stream.write(canonical(row) + '\n')

    def close(self):
        self.connection.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        self.connection.close()


def expected_keys(cfg, selected, jobs):
    result = {}
    for job in jobs:
        task = job['name']
        keys = []
        if job['kind'] == 'quality':
            for kind in ('quality', 'controlled', 'triviaqa'):
                keys.extend(f'{task}/{kind}/{case["case_id"]}/{method}'
                            for case in selected[kind] for method in cfg['quality']['methods'])
        elif job['kind'] == 'ttft':
            keys.extend(f'{task}/{case["case_id"]}/{bw}/{repeat}/{method}'
                        for case in selected['ttft'] for bw in cfg['ttft']['bandwidth_gbps']
                        for repeat in range(1, cfg['ttft']['repeats'] + 1) for method in cfg['ttft']['methods'])
        else:
            keys.extend(f'{task}/{case["case_id"]}/{trace}/{repeat}/{strategy}'
                        for case in selected['adaptive'] for trace in range(cfg['adaptive']['traces'])
                        for repeat in range(1, cfg['adaptive']['repeats'] + 1) for strategy in cfg['adaptive']['strategies'])
        if len(keys) != len(set(keys)):
            raise ValueError('Experiment matrix generated duplicate keys')
        result[task] = keys
    return result


def validate_task_records(journal, job, expected):
    """A full key matrix must also contain the promised measurements and outputs."""
    count = journal.validate(expected, job['name'])
    for row in journal.rows(job['name']):
        numeric = ['offline_prefill_seconds']
        if job['kind'] == 'adaptive':
            numeric += ['offline_all_levels_encode_seconds', 'offline_preparation_wall_seconds',
                        'transfer_seconds', 'reconstruct_seconds', 'remaining_input_seconds', 'ttft_seconds', 'answer_latency_seconds']
            decisions = row.get('decisions', [])
            if not decisions or len(decisions) != len(row.get('levels', [])):
                raise ValueError('Incomplete adaptive chunk decisions')
            previous = 0
            for decision, level in zip(decisions, row['levels']):
                if decision['start'] != previous or decision['end'] <= previous or decision['cache_tokens_after'] != decision['end'] or decision['level'] != level:
                    raise ValueError('Adaptive cache positions or tail are inconsistent')
                previous = decision['end']
            if previous != row['context_tokens']:
                raise ValueError('Adaptive context tail is missing')
        else:
            numeric += ['offline_encode_seconds']
            if job['kind'] == 'ttft':
                numeric += ['transfer_seconds', 'reconstruct_seconds', 'remaining_input_seconds', 'ttft_wall_seconds', 'actual_bandwidth_gbps']
            elif row['kind'] == 'quality':
                if row.get('tokens_scored') != len(row.get('scored_ids', [])) or row['tokens_scored'] != len(row.get('token_nll', [])) or not row['tokens_scored']:
                    raise ValueError('Incomplete continuation token scores')
        for field in numeric:
            value = row.get(field)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f'Missing/invalid {field} in {row["record_id"]}')
        if row['kind'] in ('controlled', 'triviaqa', 'adaptive'):
            if not isinstance(row.get('prediction'), str) or not isinstance(row.get('generated_ids'), list):
                raise ValueError('QA generation evidence is missing')
            if any(type(row.get(field)) not in (int, float) or not 0 <= row[field] <= 1 for field in ('em', 'f1')):
                raise ValueError('QA scores are missing/invalid')
    return count
