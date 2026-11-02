import unittest
from experiments.joint_profiles import (fit_residuals, verify_residuals, RuntimeProfile,
                                       nearest_rank, BandwidthHistory, validate_source_split)


def records():
    return [dict(request_id=str(i), source_id=str(i % 5), execution_mode='pipeline',
                 route='chunks', total_context_tokens=1024, cache_class='cold',
                 F_base_ns=1000000000, first_token_ns=1000000000 + i * 1000000,
                 nll_delta=.01, em_retention=1.0, success=True)
            for i in range(20)]


class CalibrationTests(unittest.TestCase):
    def test_request_max_before_nearest_rank(self):
        rows = records()
        rows.append(dict(rows[0], F_base_ns=0))
        table = fit_residuals(rows)
        key, delta = table.lookup(('pipeline', 'chunks', 1024, 'cold'))
        self.assertEqual(key, ('pipeline', 'chunks', 1024, 'cold'))
        self.assertAlmostEqual(delta, .018)
        self.assertEqual(nearest_rank(range(1, 21), .9), 18)

    def test_inadequate_group_falls_back_to_real_global_table(self):
        rows = records()
        rows[0]['cache_class'] = 'mixed'
        table = fit_residuals(rows)
        key, delta = table.lookup(('pipeline', 'chunks', 1024, 'mixed'))
        self.assertEqual(key, ('pipeline', 'chunks', 1024))
        self.assertAlmostEqual(delta, .017)

    def test_coverage_is_request_level_and_failures_fail_acceptance(self):
        rows = records()
        table = fit_residuals(rows)
        self.assertTrue(verify_residuals(rows, table)['ready'])
        rows.append(dict(rows[0], success=False))
        self.assertFalse(verify_residuals(rows, table)['ready'])

    def test_source_partitions_are_disjoint_and_complete(self):
        split = {k: list(range(i * 5, (i + 1) * 5)) for i, k in enumerate(('cost', 'residual', 'verify', 'test'))}
        validate_source_split(split)
        split['test'][0] = 0
        with self.assertRaisesRegex(ValueError, 'overlap'):
            validate_source_split(split)

    def test_bandwidth_uses_completed_large_messages_and_valid_history(self):
        b = BandwidthHistory()
        self.assertEqual(b.beta50_bps, 1e9)
        b.observe(100, 0, 100)
        self.assertFalse(b.valid)
        b.observe(262144, 0, 2097152)
        self.assertTrue(b.valid)
        self.assertEqual(b.beta50_bps, 1e9)
        b.observe(262144, 0, 4194304)
        self.assertEqual(b.beta50_bps, .875e9)
        self.assertEqual(b.beta90_bps, .5e9)
        b.observe(262144, 1, 0)
        self.assertFalse(b.valid)

    def test_profiles_roundtrip_and_synthetic_rejected_by_formal_gate(self):
        p = RuntimeProfile(fingerprint='abc', synthetic=True, ready=True)
        self.assertEqual(RuntimeProfile.from_dict(p.to_dict()).fingerprint, 'abc')
        with self.assertRaisesRegex(ValueError, 'synthetic'):
            p.require_formal()

    def test_one_missed_decision_makes_request_uncovered_once(self):
        rows = records()
        table = fit_residuals(rows)
        rows.append(dict(rows[0], F_safe_ns=0))
        report = verify_residuals(rows, table)
        self.assertEqual(report['groups'][0]['request_count'], 20)
        self.assertAlmostEqual(report['groups'][0]['coverage'], .85)
        self.assertFalse(report['ready'])

    def test_required_matrix_missing_groups_fail(self):
        rows = records()
        report = verify_residuals(rows, fit_residuals(rows),
                                  required_groups=[('pipeline', 'whole_text', 1024, 'cold')])
        self.assertFalse(report['ready'])
        self.assertEqual(len(report['missing_groups']), 1)

    def test_warm_history_rejects_future_stale_and_other_links(self):
        b = BandwidthHistory(link_id='a')
        b.import_completed([dict(size=262144, start_ns=0, end_ns=2097152, link_id='a'),
                            dict(size=262144, start_ns=6_000_000_000, end_ns=6_002_097_152, link_id='b')],
                           6_003_000_000)
        self.assertFalse(b.valid)


if __name__ == '__main__':
    unittest.main()
