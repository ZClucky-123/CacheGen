"""Durable benchmark observations and separate diagnostic/formal recovery."""
import json
from pathlib import Path
import time

from .joint_calibration import append_record, save_json


def benchmark_directory(run, diagnostic_only=False):
    return Path(run) / ('diagnostic-benchmark' if diagnostic_only else 'benchmark')


def _validate(row, diagnostic_only):
    if not isinstance(row, dict) or not isinstance(row.get('key'), str) or not row['key']:
        raise ValueError('benchmark_record_key_missing')
    if row.get('diagnostic_only', False) is not diagnostic_only:
        raise ValueError('benchmark_record_scope_mismatch')
    if row.get('synthetic', False) and not diagnostic_only:
        raise ValueError('benchmark_record_synthetic_in_formal_scope')


def iter_benchmark_records(path, diagnostic_only=False, *, repair_tail=False):
    """Yield unique complete rows. Keep offsets, rather than all wire/event data.

    Only a newline-terminated record is committed. An incomplete tail may be
    preserved and removed during recovery. Corrupt complete rows and conflicting
    repeated keys require explicit investigation instead of silent weighting.
    """
    path = Path(path)
    if not path.exists():
        return
    offsets = {}
    partial = None
    with path.open('rb') as stream:
        while True:
            start = stream.tell()
            line = stream.readline()
            if not line:
                break
            if not line.endswith(b'\n'):
                if not repair_tail:
                    raise ValueError('benchmark_record_truncated_tail')
                partial = (start, line)
                break
            if not line.strip():
                continue
            try:
                row = json.loads(line.decode('utf-8'))
            except (ValueError, UnicodeError) as error:
                raise ValueError(f'benchmark_record_corrupt at byte {start}') from error
            _validate(row, diagnostic_only)
            key = row['key']
            if key in offsets:
                current = stream.tell()
                stream.seek(offsets[key])
                previous = json.loads(stream.readline().decode('utf-8'))
                stream.seek(current)
                if row != previous:
                    raise ValueError('benchmark_record_conflicting_duplicate: ' + key)
                continue
            offsets[key] = start
            yield row
    if partial is not None:
        start, content = partial
        backup = path.with_name(path.name + '.partial-' + str(time.perf_counter_ns()))
        backup.write_bytes(content)
        with path.open('r+b') as stream:
            stream.truncate(start)


def read_benchmark_records(path, diagnostic_only=False):
    return list(iter_benchmark_records(path, diagnostic_only))


def load_benchmark_checkpoint(output, diagnostic_only=False):
    output = Path(output)
    done = {row['key'] for row in iter_benchmark_records(
        output / 'records.jsonl', diagnostic_only, repair_tail=True)}
    checkpoint = output / 'checkpoint.json'
    saved = json.loads(checkpoint.read_text(encoding='utf-8')) if checkpoint.exists() else []
    if not isinstance(saved, list) or any(not isinstance(key, str) for key in saved):
        raise ValueError('benchmark_checkpoint_invalid')
    if set(saved) - done:
        raise ValueError('benchmark_checkpoint_missing_observation')
    if set(saved) != done:
        save_json(checkpoint, sorted(done))
    return done


def record_benchmark_completion(output, row, done, diagnostic_only=False):
    _validate(row, diagnostic_only)
    if row['key'] in done:
        return False
    output = Path(output)
    append_record(output / 'records.jsonl', row, durable=True)
    done.add(row['key'])
    save_json(output / 'checkpoint.json', sorted(done))
    return True
