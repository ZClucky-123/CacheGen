import unittest
from experiments.joint_types import begin_request
from experiments.joint_cache import PacketCache
from experiments.joint_source import SourceStore


class CacheTests(unittest.TestCase):
    def test_clock_and_negative_budget(self):
        tick = [10]
        request = begin_request(.000000002, now=lambda: tick[0])
        self.assertEqual(request.deadline_ns, 12)
        tick[0] = 15
        self.assertLess(request.remaining_s(), 0)

    def test_pinned_cache_uses_temporary_result_and_evicts(self):
        cache = PacketCache(4)
        cache.put(('a',), b'abcd')
        with cache.acquire(('a',)) as value:
            self.assertEqual(value, b'abcd')
            self.assertFalse(cache.put(('b',), b'xy'))
            self.assertEqual(cache.resident_bytes, 4)
        self.assertTrue(cache.put(('b',), b'xy'))
        self.assertIsNone(cache.peek(('a',)))

    def test_source_release_waits_for_reader_and_revision_changes(self):
        store = SourceStore()
        ref = store.register_context([1, 2], object(), 'r1')
        with store.lease(ref.context_id, 'r1') as raw:
            store.release_master(ref.context_id, wait=False)
            self.assertIsNotNone(raw)
            self.assertFalse(store.available(ref.context_id))
        self.assertIsNone(store.get(ref.context_id).raw_kv)
        with self.assertRaisesRegex(RuntimeError, 'source_unavailable'):
            with store.lease(ref.context_id, 'r1'):
                pass

    def test_same_key_generation_is_coalesced(self):
        cache = PacketCache(16)
        first, owner = cache.reserve(('a',))
        second, duplicate = cache.reserve(('a',))
        self.assertTrue(owner)
        self.assertFalse(duplicate)
        self.assertIs(first, second)
        cache.complete(('a',), b'ok')
        self.assertEqual(second.result(), b'ok')
