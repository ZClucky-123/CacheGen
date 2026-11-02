import unittest
from dataclasses import replace

from experiments.joint_scheduler import CandidateEstimate, Plan, Snapshot, RunningTask, CacheEntry, simulate_plan
from experiments.joint_profiles import RuntimeProfile


def item(i=0, q=2, encoding='fixed_bits', size=100, g=10, c=2, r=5, ready=False):
    return CandidateEstimate(i, i * 256, (i + 1) * 256, q, encoding, (i, q, encoding),
                             size, size, g, g, c, c, r, r,
                             packet_state='ready' if ready else 'missing')


class SimulatorTests(unittest.TestCase):
    def test_serial_includes_all_generation_seal_network_and_restore(self):
        plan = Plan(2, (item(), item(1)))
        result = simulate_plan(plan, Snapshot(100, mode='serial', beta50_bps=80e9))
        self.assertEqual(result.end_ns, 100 + 2 * (10 + 2 + 10 + 5))
        self.assertEqual(result.generation_count, 2)

    def test_inflight_encoding_is_not_preempted_by_arriving_restore(self):
        first, second = item(0, ready=True), item(1, g=30)
        snapshot = Snapshot(0, history_valid=True, beta50_bps=80e9)
        result = simulate_plan(Plan(2, (first, second)), snapshot)
        events = {(kind, block): t for t, kind, block in result.events}
        self.assertEqual(events['NET_START', 0], 0)
        self.assertEqual(events['ENCODE_START', 1], 0)
        self.assertEqual(events['DATA_AVAILABLE', 0], 10)
        self.assertEqual(events['RESTORE_START', 0], 30)

    def test_ready_packets_reduce_to_bounded_network_restore_pipeline(self):
        result = simulate_plan(Plan(2, (item(0, r=30, ready=True), item(1, r=30, ready=True),
                                      item(2, r=30, ready=True))),
                               Snapshot(0, history_valid=True, beta50_bps=80e9))
        self.assertEqual(result.end_ns, 100)
        self.assertEqual(result.generation_count, 0)
        self.assertEqual(result.cache_class, 'warm')

    def test_inflight_remaining_time_charged_once(self):
        a = item(0)
        result = simulate_plan(Plan(2, (a,)), Snapshot(100, beta50_bps=80e9,
                               inflight=(RunningTask('gpu', 'encode', 0, 7, 9, a),)))
        self.assertEqual(result.end_ns, 124)

    def test_no_history_defers_next_selection_until_completed_sample(self):
        result = simulate_plan(Plan(2, (item(0, ready=True), item(1))),
                               Snapshot(0, history_valid=False, beta50_bps=80e9))
        events = {(kind, block): t for t, kind, block in result.events}
        self.assertGreaterEqual(events['SELECT_DONE', 1], events['DATA_AVAILABLE', 0])

    def test_snapshot_only_holds_metadata_and_is_not_mutated(self):
        a = item()
        snap = Snapshot(0, cache_capacity=50)
        result = simulate_plan(Plan(2, (a,)), snap)
        self.assertEqual(snap.cache_entries, ())
        self.assertEqual(result.temporary_peak_bytes, 100)

    def test_running_network_without_history_does_not_start_next_encode_early(self):
        a, b = item(0, ready=True), item(1)
        snap = Snapshot(100, beta50_bps=80e9,
                        inflight=(RunningTask('network', 'network', 0, 7, 7, a),))
        result = simulate_plan(Plan(2, (b,)), snap)
        events = {(kind, block): t for t, kind, block in result.events}
        self.assertEqual(events['DATA_AVAILABLE', 0], 107)
        self.assertGreaterEqual(events['SELECT_DONE', 1], 107)
        self.assertEqual(result.generation_count, 1)

    def test_received_prefix_has_restore_priority_over_queued_encode(self):
        a, b = item(0, ready=True), item(1)
        snap = Snapshot(0, history_valid=True, beta50_bps=80e9,
                        stages=((a, 'received'), (b, 'encode_queued')))
        result = simulate_plan(Plan(2, (b,)), snap)
        events = {(kind, block): t for t, kind, block in result.events}
        self.assertEqual(events['RESTORE_START', 0], 0)
        self.assertEqual(events['ENCODE_START', 1], 5)

    def test_evicted_future_cache_hit_is_generated_again(self):
        first, future = item(0), item(1, ready=True)
        snap = Snapshot(0, mode='serial', beta50_bps=80e9, cache_capacity=100,
                        cache_entries=(CacheEntry(future.cache_key, 100),))
        result = simulate_plan(Plan(2, (first, future)), snap)
        self.assertIn(future.cache_key, result.evictions)
        self.assertEqual(result.generation_count, 2)
        self.assertEqual(result.cache_class, 'cold')

    def test_profile_transition_override_preloaded_once(self):
        a = item(0)
        profile = RuntimeProfile(costs=[dict(q=2, encoding='fixed_bits', N=256,
                   previous_gpu='none', buffer_state='first', mode='serial', prefix=0,
                   g50_s=30e-9, g90_s=30e-9)])
        result = simulate_plan(Plan(2, (a,)), Snapshot(0, mode='serial', beta50_bps=80e9), profile)
        self.assertEqual(result.end_ns, 47)

    def test_zero_cost_tasks_finish_without_deadlock(self):
        a = item(g=0, c=0, r=0, size=0)
        result = simulate_plan(Plan(2, (a,)), Snapshot(123))
        self.assertEqual(result.end_ns, 123)

    def test_old_inflight_generation_pays_seal_without_sending_old_wire(self):
        old = item(0, encoding='ac_count', c=20)
        current = item(0, encoding='fixed_bits', g=5, c=2, r=3)
        snap = Snapshot(0, beta50_bps=80e9,
                        inflight=(RunningTask('gpu', 'encode', 0, 7, 7, old),))
        result = simulate_plan(Plan(2, (current,)), snap)
        events = {(kind, block): t for t, kind, block in result.events}
        self.assertEqual(events['ENCODE_START', 0], 27)
        self.assertEqual(events['NET_START', 0], 34)
        self.assertEqual(result.end_ns, 47)
        self.assertEqual(result.generation_count, 1)

    def test_evicted_ready_packet_with_released_source_is_unavailable(self):
        first, future = item(0), replace(item(1, ready=True), source_available=False)
        snap = Snapshot(0, mode='serial', beta50_bps=80e9, cache_capacity=100,
                        cache_entries=(CacheEntry(future.cache_key, 100),))
        with self.assertRaisesRegex(ValueError, 'source_unavailable'):
            simulate_plan(Plan(2,(first,future)),snap)

    def test_evicted_ready_packet_without_generation_cost_is_unavailable(self):
        first, future = item(0), replace(item(1, ready=True), g50_ns=0,g90_ns=0,
                                        c50_ns=0,c90_ns=0,generation_cost_available=False)
        snap = Snapshot(0, mode='serial', beta50_bps=80e9, cache_capacity=100,
                        cache_entries=(CacheEntry(future.cache_key, 100),))
        with self.assertRaisesRegex(ValueError, 'cost_model_missing'):
            simulate_plan(Plan(2,(first,future)),snap)


if __name__ == '__main__':
    unittest.main()
