import json
import tempfile
import unittest
from pathlib import Path

from experiments.joint_config import REPO_ROOT, load_joint_config, run_directory
from experiments.joint_calibration import source_partition, manifest_for, validate_partition, append_record
from experiments.joint_calibration import summarize, calibration_states, stage_guard
from experiments.joint_calibration import (frozen_artifact_digests, verify_frozen_artifacts,
    invalidate_freeze, save_json, quality_conditions)
from experiments.joint import seed_request_options, benchmark_condition_supported


class JointCliTests(unittest.TestCase):
    def test_quality_change_seed_uses_other_admitted_quality_only(self):
        self.assertEqual(seed_request_options({'fixed_quality':2,'encodings':['fixed_bits']},'quality_change',4096),
                         {'quality_mode':'fixed','fixed_quality':3,'encodings':['fixed_bits']})
        self.assertEqual(seed_request_options({'fixed_quality':3,'encodings':['ac_count']},'quality_change',4096)['fixed_quality'],2)
        self.assertEqual(seed_request_options({'fixed_quality':1,'original':'cachegen1'},'quality_change',1024)['original'],'cachegen2')
        self.assertFalse(benchmark_condition_supported({'name':'joint-adaptive'},'pipeline','quality_change'))

    def test_representation_miss_seed_is_same_quality_outside_target_set(self):
        target={'fixed_quality':2,'encodings':['ac_count','fixed_bits']}
        seed=seed_request_options(target,'representation_miss',4096)
        self.assertEqual(seed,{'quality_mode':'fixed','fixed_quality':2,'encodings':['ac_cdf']})
        self.assertFalse(set(seed['encodings']).intersection(target['encodings']))
        self.assertFalse(benchmark_condition_supported({'fixed_quality':2},'pipeline','representation_miss'))
        self.assertFalse(benchmark_condition_supported({'fixed_quality':2,'original':'cachegen2'},'serial','representation_miss'))
    def test_default_output_independent_of_working_directory(self):
        self.assertEqual(run_directory(run_id='cpu-check'), REPO_ROOT / 'optimization/output/cpu-check')
        with self.assertRaises(ValueError):
            run_directory('results/new')
        with self.assertRaises(ValueError):
            run_directory(run_id='../escape')

    def test_twenty_independent_sources_preserve_original_test_five(self):
        split = source_partition(list(range(25)), [0, 1, 2, 3, 4])
        self.assertEqual(split['cost'], [5, 6, 7, 8, 9])
        self.assertEqual(split['residual'], [10, 11, 12, 13, 14])
        self.assertEqual(split['verification'], [15, 16, 17, 18, 19])
        validate_partition(split)
        with self.assertRaises(ValueError):
            source_partition(list(range(19)), list(range(5)))
        split['cost'][0] = 0
        with self.assertRaises(ValueError):
            validate_partition(split)

    def test_manifest_describes_recipes_without_claiming_packets(self):
        manifest = manifest_for({'source_id': 5, 'context_ids': [1] * 513}, [1, 2, 3], 256)
        self.assertEqual(len(manifest['blocks']), 3)
        self.assertEqual(manifest['blocks'][-1]['end'], 513)
        self.assertEqual(manifest['blocks'][0]['metadata_candidates'], 9)
        self.assertFalse(manifest['packets_persisted'])

    def test_default_configuration_is_v22(self):
        cfg = load_joint_config()
        self.assertEqual(cfg['protocol'], 'cachegen-joint-v2')
        self.assertEqual(cfg['receive_capacity'], 1)
        self.assertEqual(cfg['generation_lookahead'], 1)
        self.assertEqual(cfg['packet_cache_bytes_per_context'], 268435456)

    def test_record_writer_preserves_actual_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'records.jsonl'
            append_record(target, {'request_id': 'r', 'first_token_ns': 999})
            self.assertEqual(json.loads(target.read_text())['first_token_ns'], 999)

    def test_calibration_state_keys_include_fallback_and_resource_state(self):
        cfg = load_joint_config()
        states = list(calibration_states(cfg))
        self.assertTrue(any(s['tag'] == 0x11 for s in states))
        self.assertEqual({s['previous_gpu'] for s in states}, {'none', 'initialize', 'encode', 'restore'})
        self.assertEqual({s['buffer_state'] for s in states if s['encoding'] not in ('text','whole_text')}, {'first', 'reuse', 'expand'})
        self.assertTrue(all(s['buffer_state'] == 'allocates_each_restore' for s in states if s['encoding'] in ('text','whole_text')))
        self.assertTrue(all(s['tag'] == 0x10 for s in states if s['encoding'] == 'ac_cdf'))
        self.assertEqual(summarize(list(range(1, 21))), (10.5, 18))

    def test_formal_benchmark_rejects_non_ready_or_synthetic_artifact(self):
        with self.assertRaisesRegex(ValueError, 'ready'):
            stage_guard({'state': 'costs'}, 'benchmark')
        with self.assertRaisesRegex(ValueError, 'synthetic'):
            stage_guard({'state': 'ready', 'synthetic': True}, 'benchmark')

    def test_frozen_profile_mutation_and_recalibration_invalidate_freeze(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)
            for name in ('runtime_profile.json','cost_rows.json','first_rows.json','residual-profile.json',
                         'verification.json','source-split.json','joint-config.json','base-config.json'):
                save_json(run/name, {})
            save_json(run/'frozen.json', {'fingerprint':'f','artifacts':frozen_artifact_digests(run)})
            manifest={'frozen':True}
            verify_frozen_artifacts(run,'f',manifest)
            save_json(run/'cost_rows.json', {'modified':True})
            with self.assertRaisesRegex(ValueError,'frozen_artifact_mismatch'):
                verify_frozen_artifacts(run,'f',manifest)
            invalidate_freeze(run,manifest)
            self.assertFalse(manifest['frozen'])
            self.assertFalse((run/'frozen.json').exists())

    def test_qa_retention_is_condition_specific_and_requires_positive_baseline(self):
        rows=[]
        for condition in ('cold','warm','mixed'):
            for source in range(5):
                for repeat in range(4):
                    rows.append({'condition':['pipeline','chunks',1024,condition,'stable'],
                        'source_id':source,'raw_em':0. if condition=='mixed' else 1.,
                        'joint_em':1. if condition=='cold' else 0.})
        reports={r['condition'][3]:r for r in quality_conditions(rows)}
        self.assertTrue(reports['cold']['passed'])
        self.assertFalse(reports['warm']['passed'])
        self.assertFalse(reports['mixed']['passed'])


if __name__ == '__main__':
    unittest.main()
