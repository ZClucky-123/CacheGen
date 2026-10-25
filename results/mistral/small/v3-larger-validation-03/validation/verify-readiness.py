"""Read-only evidence checks, then archive this readiness validation evidence."""
import collections
import hashlib
import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from experiments.artifacts import code_data_identity, seal, verify
from experiments.records import validate_task_records

ROOT = Path('/home/aaa/CacheGen')
RUN = ROOT / 'results/mistral/small/v3-larger-validation-03'
read = lambda p: json.loads(p.read_text())
manifest = read(RUN / 'manifest.json')
assert manifest['status'] == 'completed' and manifest['exit_code'] == 0
launch = read(ROOT / '.local/launches/mistral-v3-readiness-03-resume-2.json')
assert launch['status'] == 'completed' and launch['exit_code'] == 0
base_archive_files = verify(RUN)
cfg = read(RUN / 'config.json')
assert code_data_identity(ROOT, cfg) == manifest['identity']['inputs']
expected = read(RUN / 'expected.json')
rows = [json.loads(line) for line in (RUN / 'records.jsonl').read_text().splitlines()]
class ReadOnlyJournal:
    def rows(self, task=None):
        return [r for r in rows if task is None or r['task'] == task]
    def validate(self, keys, task=None):
        actual = [r['record_id'] for r in self.rows(task)]
        assert len(actual) == len(set(actual)) and set(actual) == set(keys)
        return len(actual)
journal = ReadOnlyJournal()
counts = {j['name']: validate_task_records(journal, j, expected[j['name']]) for j in manifest['jobs']}
assert sum(counts.values()) == len(rows) == 469
checks = read(RUN / 'cache_checks.json')
assert checks['status'] == 'passed' and len(checks['cases']) == 15
assert all(c['text_chunk_kv_exact'] and c['text_chunk_kv_audit']['elementwise_allclose'] for c in checks['cases'])
assert len(checks['complete_nll_checks']) == 6
assert all(c['absolute_error'] <= .01 and c['tokens_scored'] == 32 for c in checks['complete_nll_checks'])
transport_checks = read(RUN / 'ttft_checks.json')
assert all(c['byte_identity'] and c['mode_kv_identity'] for c in transport_checks)
assert all(c['text_vs_whole_kv_audit']['elementwise_allclose'] for c in transport_checks if c['method'] == 'text')
proof = read(ROOT / '.local/larger-readiness-recovery-before.json')
assert proof['status'] == 'interrupted_verified' and proof['target_tokens'] == 4096
assert proof['initial_launch']['exit_code'] == 143
with sqlite3.connect(f'file:{RUN / "records.sqlite3"}?mode=ro', uri=True) as db:
    current_hashes = dict(db.execute('SELECT key, sha256 FROM observations'))
assert all(current_hashes.get(k) == v for k, v in proof['rows_before_resume'].items())
case_id = proof['case_id']
prepared_matches = 0
for name, prepared in proof['prepared_before_resume'].items():
    path = Path(name)
    size, method = int(path.parts[-3]), path.parts[-2]
    matches = [r for r in rows if r['task'] == 'ttft' and r['case_id'] == case_id and r['chunk_size'] == size and r['method'] == method]
    assert matches
    assert all(r['prepared_payload_sha256'] == [p['sha256'] for p in prepared['packets']] for r in matches)
    prepared_matches += len(matches)
pairs = collections.defaultdict(dict)
for row in rows:
    if row['task'] == 'ttft' and row['mode'] in ('chunk_serial', 'chunk_pipeline'):
        group = tuple(row[k] for k in ('case_id', 'bandwidth_gbps', 'repetition', 'method', 'chunk_size', 'queue_capacity'))
        pairs[group][row['mode']] = row
cross_attempt_pairs = 0
for pair in pairs.values():
    a, b = pair['chunk_serial'], pair['chunk_pipeline']
    assert a['prepared_payload_sha256'] == b['prepared_payload_sha256']
    assert a['bytes'] == b['bytes'] and a['first_token_id'] == b['first_token_id']
    cross_attempt_pairs += a['attempt'] != b['attempt']
assert cross_attempt_pairs > 0
assert len(manifest['attempts']) == 3
assert [a.get('exit_code') for a in manifest['attempts']] == [143, None, 0]
hard_restart = read(ROOT / '.local/larger-readiness-hard-restart.json')
assert len(hard_restart['rows_preserved']) == 441
assert all(current_hashes.get(k) == v for k, v in hard_restart['rows_preserved'].items())
failures = [json.loads(line) for line in (RUN / 'failures.jsonl').read_text().splitlines()]
assert len(failures) == 1 and failures[0]['error'] == 'KeyboardInterrupt(15)'
assert not any((RUN / 'pending-prepared').rglob('prepared.json'))
out = RUN / 'validation'
out.mkdir(exist_ok=True)
for src, dest in [('larger-readiness-tests.log', '111-tests.log'), ('larger-compatibility-tests.log', '3-compatibility-tests.log'),
                  ('larger-readiness-recovery.py', 'recovery-controller.py'), ('larger-readiness-recovery-before.json', 'recovery-before.json'),
                  ('larger-readiness-hard-restart.json', 'environment-restart.json')]:
    shutil.copy2(ROOT / '.local' / src, out / dest)
result = {'status': 'passed', 'validated_utc': datetime.now(timezone.utc).isoformat(),
          'formal_validation_observations': len(rows), 'task_counts': counts, 'unit_tests': 111, 'compatibility_tests': 3,
          'strict_text_kv_exact_cases': 15, 'full_continuation_nll_checks': 6,
          'transport_correctness_checks_including_resumed_recheck': len(transport_checks),
          'unchanged_observations_after_resume': len(proof['rows_before_resume']),
          'persisted_representation_count': len(proof['prepared_before_resume']),
          'formal_rows_matching_persisted_representation': prepared_matches,
          'serial_pipeline_pairs': len(pairs), 'cross_attempt_identical_payload_pairs': cross_attempt_pairs,
          'attempt_exit_codes': [143, None, 0], 'hard_restart_preserved_rows': 441,
          'unknown_exit_note': 'Attempt 2 lost its process environment, so no exit code was written; not classified as a clean exit.',
          'base_archive_verified_files': base_archive_files,
          'peak_gpu_allocated_bytes': manifest['peak_gpu_allocated_bytes'],
          'peak_gpu_reserved_bytes': manifest['peak_gpu_reserved_bytes'],
          'peak_host_rss_bytes': manifest['peak_host_rss_bytes'], 'source_files': len(manifest['identity']['inputs']['code'])}
(out / 'validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
shutil.copy2(__file__, out / 'verify-readiness.py')
seal(RUN)
result['verified_archive_files'] = verify(RUN)
(ROOT / '.local/larger-readiness-validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(result, ensure_ascii=False, indent=2))
