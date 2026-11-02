"""Crash recovery and diagnostic/formal record isolation without GPU work."""
import contextlib
import csv
import io
import json
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments.joint_calibration import append_record, save_json
from experiments.joint_records import (benchmark_directory, load_benchmark_checkpoint,
    read_benchmark_records, record_benchmark_completion)


class JointRecordTests(unittest.TestCase):
    def test_append_before_checkpoint_crash_recovers_completed_observation(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'benchmark'
            done = load_benchmark_checkpoint(output)
            row = {'key': 'a', 'diagnostic_only': False, 'ttft_s': 1.}
            with patch('experiments.joint_records.save_json', side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    record_benchmark_completion(output, row, done)
            recovered = load_benchmark_checkpoint(output)
            self.assertEqual(recovered, {'a'})
            self.assertFalse(record_benchmark_completion(output, row, recovered))
            self.assertEqual(read_benchmark_records(output / 'records.jsonl'), [row])
            self.assertEqual(json.loads((output / 'checkpoint.json').read_text()), ['a'])

    def test_diagnostic_and_formal_use_separate_directories_and_completion_sets(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            diagnostic = benchmark_directory(root, True)
            formal = benchmark_directory(root, False)
            self.assertNotEqual(diagnostic, formal)
            record_benchmark_completion(diagnostic, {'key': 'a', 'diagnostic_only': True}, set(), True)
            self.assertEqual(load_benchmark_checkpoint(formal), set())
            record_benchmark_completion(formal, {'key': 'a', 'diagnostic_only': False}, set())
            self.assertEqual(load_benchmark_checkpoint(diagnostic, True), {'a'})
            self.assertEqual(load_benchmark_checkpoint(formal), {'a'})

    def test_partial_tail_is_preserved_then_removed_before_append(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            row = {'key': 'a', 'diagnostic_only': False}
            good = (json.dumps(row) + '\n').encode()
            partial = b'{"key":"incomplete","text":"\xe4\xb8'
            (output / 'records.jsonl').write_bytes(good + partial)
            done = load_benchmark_checkpoint(output)
            self.assertEqual(done, {'a'})
            backups = list(output.glob('records.jsonl.partial-*'))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), partial)
            self.assertEqual((output / 'records.jsonl').read_bytes(), good)
            record_benchmark_completion(output, {'key': 'b', 'diagnostic_only': False}, done)
            self.assertEqual([r['key'] for r in read_benchmark_records(output / 'records.jsonl')], ['a', 'b'])

    def test_corrupt_complete_record_is_rejected_without_rewriting(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            content = b'{"key":"a"}\ninvalid json\n{"key":"b"}\n'
            (output / 'records.jsonl').write_bytes(content)
            with self.assertRaisesRegex(ValueError, 'corrupt'):
                load_benchmark_checkpoint(output)
            self.assertEqual((output / 'records.jsonl').read_bytes(), content)

    def test_identical_duplicates_are_counted_once_and_conflicts_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'records.jsonl'
            row = {'key': 'a', 'diagnostic_only': False, 'ttft_s': 1.}
            append_record(path, row)
            append_record(path, row)
            self.assertEqual(read_benchmark_records(path), [row])
            append_record(path, dict(row, ttft_s=2.))
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                read_benchmark_records(path)

    def test_checkpoint_cannot_skip_a_missing_record(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            save_json(output / 'checkpoint.json', ['missing'])
            with self.assertRaisesRegex(ValueError, 'checkpoint'):
                load_benchmark_checkpoint(output)

    def test_formal_reader_rejects_diagnostic_records(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'records.jsonl'
            append_record(path, {'key': 'a', 'diagnostic_only': True})
            with self.assertRaisesRegex(ValueError, 'scope'):
                read_benchmark_records(path)

    def test_analysis_does_not_duplicate_weight_and_keeps_diagnostic_outputs_separate(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            output = benchmark_directory(run, True)
            common = dict(T=1024, execution_mode='serial', target_gbps=1., slo_s=2.,
                          cache_condition='cold', source_id=0, repeat=0, slo_met=True,
                          diagnostic_only=True)
            baseline = dict(common, key='baseline', group='original-q2', ttft_s=2.)
            treatment = dict(common, key='treatment', group='fixed_bits-q2', ttft_s=1.)
            for row in (baseline, treatment, treatment):
                append_record(output / 'records.jsonl', row)
            script = Path(__file__).resolve().parents[1] / 'optimization/scripts/analyze.py'
            with patch('experiments.joint_config.run_directory', return_value=run), \
                 patch.object(sys, 'argv', [str(script), '--run-dir', str(run), '--diagnostic-only']), \
                 contextlib.redirect_stdout(io.StringIO()):
                runpy.run_path(str(script), run_name='__main__')
            with (output / 'summary.csv').open(encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream))
            self.assertTrue(all(row['requests'] == '1' for row in rows))
            self.assertFalse((run / 'benchmark/summary.csv').exists())
            self.assertIn('diagnostic', (output / 'report.md').read_text())
