"""Durable per-observation checkpoints and strict matrix completeness."""
import hashlib
import json
import sqlite3
import math
from datetime import datetime, timezone
from pathlib import Path
from experiments.config import TASK_KINDS, ttft_variants, ablation_cases


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
            keys.extend(ttft_key(task, case['case_id'], bw, repeat, method, mode, size, cfg['transport']['queue_capacity'])
                        for case in selected['ttft'] for bw in cfg['ttft']['bandwidth_gbps']
                        for repeat in range(1, cfg['ttft']['repeats'] + 1) for method in cfg['ttft']['methods']
                        for mode, size in ttft_variants(cfg, case.get('context_tokens', cfg['ttft']['lengths'][0])))
        elif job['kind'] == 'adaptive':
            keys.extend(adaptive_key(task, case['case_id'], trace, repeat, strategy, cfg)
                        for case in selected['adaptive'] for trace in range(cfg['adaptive']['traces'])
                        for repeat in range(1, cfg['adaptive']['repeats'] + 1) for strategy in cfg['adaptive']['strategies'])
        elif job['kind'] == 'ablation':
            keys.extend(f'{task}/{kind}/{case["case_id"]}/{precision}/{entropy}'
                        for kind, case in ablation_cases(cfg, selected)
                        for precision in cfg['ablation']['precisions'] for entropy in cfg['ablation']['entropy'])
        else:
            raise ValueError(f'Unknown task kind: {job["kind"]}')
        if len(keys) != len(set(keys)):
            raise ValueError('Experiment matrix generated duplicate keys')
        result[task] = keys
    return result


def ttft_key(task, case_id, bandwidth, repeat, method, mode, chunk_size, capacity):
    return f'{task}/{case_id}/{bandwidth}/{repeat}/{method}/{mode}/chunk-{chunk_size}/queue-{capacity}'


def adaptive_key(task, case_id, trace, repeat, strategy, cfg):
    estimator = cfg['adaptive']['throughput_estimator'] if strategy == 'adaptive_history' else 'current_trace'
    return f'{task}/{case_id}/{trace}/{repeat}/{strategy}/{estimator}/chunk-{cfg["adaptive"]["chunk_size"]}'


def validate_task_records(journal, job, expected):
    """A full key matrix must also contain the promised measurements and outputs."""
    if job['kind'] not in TASK_KINDS:
        raise ValueError(f'Unknown task kind: {job["kind"]}')
    count = journal.validate(expected, job['name'])
    for row in journal.rows(job['name']):
        value = row.get('offline_prefill_seconds')
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f'Missing/invalid offline_prefill_seconds in {row["record_id"]}')
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
                if decision.get('source_chunk_index') is not None and decision['source_chunk_index'] != decision['chunk_index'] - 1:
                    raise ValueError('History estimator read a nonprevious chunk')
                if row.get('strategy') == 'adaptive_history':
                    source = decision.get('estimate_source')
                    if source != ('fixed_initial' if decision['chunk_index'] == 0 else 'previous_completed_message'):
                        raise ValueError('Invalid history information boundary')
                for field in ('estimate_gbps', 'actual_bandwidth_gbps', 'effective_transfer_seconds', 'decision_seconds'):
                    if type(decision.get(field)) not in (int, float) or not math.isfinite(decision[field]) or decision[field] < 0:
                        raise ValueError(f'Missing adaptive {field}')
                if set(decision.get('candidate_forecasts_seconds', {})) != {'0', '1', '2', '3'}:
                    raise ValueError('Missing per-representation cost forecasts')
            if previous != row['context_tokens']:
                raise ValueError('Adaptive context tail is missing')
        else:
            numeric += ['offline_encode_seconds']
            if job['kind'] == 'ttft':
                numeric += ['transfer_seconds', 'reconstruct_seconds', 'remaining_input_seconds', 'ttft_wall_seconds', 'actual_bandwidth_gbps']
                validate_timeline(row)
            if job['kind'] == 'ablation':
                numeric += ['quantize_seconds', 'decode_seconds', 'symbol_data_bytes', 'bytes']
                if not row.get('symbols_identical') or not row.get('entropy_reconstructions_identical'):
                    raise ValueError('Missing exact entropy-path equality checks')
                if len(row.get('encode_timings_seconds', [])) != row.get('timing_repeats') or len(row.get('decode_timings_seconds', [])) != row.get('timing_repeats'):
                    raise ValueError('Incomplete ablation timing repetitions')
            if row['kind'] == 'quality' or row.get('evaluation_kind') == 'quality':
                if row.get('tokens_scored') != len(row.get('scored_ids', [])) or row['tokens_scored'] != len(row.get('token_nll', [])) or not row['tokens_scored']:
                    raise ValueError('Incomplete continuation token scores')
        for field in numeric:
            value = row.get(field)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f'Missing/invalid {field} in {row["record_id"]}')
        if row['kind'] in ('controlled', 'triviaqa', 'adaptive') or row.get('evaluation_kind') == 'controlled':
            if not isinstance(row.get('prediction'), str) or not isinstance(row.get('generated_ids'), list):
                raise ValueError('QA generation evidence is missing')
            if any(type(row.get(field)) not in (int, float) or not 0 <= row[field] <= 1 for field in ('em', 'f1')):
                raise ValueError('QA scores are missing/invalid')
    return count


def validate_timeline(row):
    chunks = row.get('chunks', [])
    if not chunks:
        raise ValueError('Missing receive/restore timeline')
    previous = 0
    for i, c in enumerate(chunks):
        if c['chunk_index'] != i or c['token_start'] != previous or c['token_end'] <= previous:
            raise ValueError('Timeline order or tail differs')
        if not 0 <= c['receive_start'] <= c['receive_end'] <= c['data_available'] <= c['restore_start'] <= c['restore_end'] <= row['ttft_wall_seconds']:
            raise ValueError('Invalid receive/GPU completion times')
        if row['mode'] == 'chunk_serial' and i and c['send_start'] < chunks[i - 1]['restore_end']:
            raise ValueError('Serial sent before previous restore completed')
        # Bounded receiving/queued lookahead: current restore + capacity ahead.
        cap = row['queue_capacity']
        if i > cap and c['send_start'] < chunks[i - cap - 1]['restore_start']:
            raise ValueError('Pipeline exceeded receiver credits')
        previous = c['token_end']
    if previous != row['context_tokens'] or sum(c['message_bytes'] for c in chunks) != row['bytes']:
        raise ValueError('Incomplete timeline bytes/context')
