"""Behavioral regressions for pre-send pacing, bounded overlap and controls."""
import copy
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from experiments.config import load_config, jobs, ttft_variants
from experiments.chunks import frame, unframe, prepare
from experiments.feedback import CompletedTransfer, history_bandwidth, history_decision
from experiments.mechanisms import pack_bits, unpack_bits
from experiments.records import expected_keys, ttft_key, adaptive_key, fingerprint
from mini_repro.transport import transfer_chunks

REPO = Path(__file__).resolve().parents[1]


class StreamingTests(unittest.TestCase):
    def test_receiver_availability_waits_for_full_budget_including_short_tail(self):
        # One short final quantum is the whole message: send-then-sleep would
        # make the receiver observe it immediately, despite a correct total time.
        payload = bytes(range(256)) * 100 + b'1234567'
        start = time.perf_counter()
        result = transfer_chunks([payload], .002, lambda _, b: b, mode='whole',
                                 chunk_bytes=65536, origin=start)
        row = result['chunks'][0]
        budget = len(payload) * 8 / (.002 * 1e9)
        self.assertGreaterEqual(row['data_available'] - row['send_start'], budget * .97)
        self.assertEqual(result['received'], [payload])
        self.assertEqual(result['value'], payload)

    def test_serial_ack_and_pipeline_capacity_waits_are_real(self):
        messages = [bytes([i]) * 1024 for i in range(5)]
        for mode in ('chunk_serial', 'chunk_pipeline'):
            seen = []
            caller = threading.get_ident()
            def restore(i, received):
                self.assertEqual(threading.get_ident(), caller)
                self.assertEqual(received, messages[i])
                seen.append(i)
                time.sleep(.025)  # Deliberately blocked caller, receiver stays CPU-only.
                return i
            result = transfer_chunks(messages, None, restore, mode=mode, queue_capacity=1)
            self.assertEqual(seen, list(range(5)))
            rows = result['chunks']
            if mode == 'chunk_serial':
                self.assertEqual(result['overlap_seconds'], 0.)
                for previous, current in zip(rows, rows[1:]):
                    self.assertGreaterEqual(current['send_start'], previous['restore_end'])
            else:
                self.assertGreater(result['overlap_seconds'], 0.)
                self.assertGreater(sum(r['capacity_wait_seconds'] for r in rows), .01)
                for i in range(2, len(rows)):
                    self.assertGreaterEqual(rows[i]['send_start'], rows[i - 2]['dequeue_time'])

    def test_restore_exception_cancels_threads_including_blocked_receiver(self):
        def fail(*_):
            raise ValueError('decode failed')
        with self.assertRaisesRegex(ValueError, 'decode failed'):
            transfer_chunks([b'x' * 100000] * 8, .01, fail, mode='chunk_pipeline', queue_capacity=1)
        self.assertFalse(any(t.name.startswith('cachegen-') for t in threading.enumerate()))

    def test_frame_counts_headers_metadata_and_detects_tail_corruption(self):
        metadata = {'chunk_index': 2, 'token_start': 256, 'token_end': 257, 'method': 'raw'}
        wire = frame(b'tail', metadata)
        meta, payload = unframe(wire)
        self.assertEqual(payload, b'tail')
        self.assertEqual(meta['payload_bytes'], 4)
        self.assertGreater(len(wire), 4)
        for invalid in (wire[:-1], wire + b'x', b'bad'):
            with self.assertRaises(ValueError):
                unframe(invalid)


class FeedbackAndMatrixTests(unittest.TestCase):
    def config(self):
        return load_config(REPO / 'experiments/configs/minimal.json', 'tinyllama')

    def test_history_initialization_and_boundary_exclude_gpu_wait(self):
        self.assertEqual(history_bandwidth([], 1.)['estimate_gbps'], 1.)
        # Only these three fields can enter the estimator; queue and GPU waits
        # are absent. The transport computes the effective interval after ready.
        completed = [CompletedTransfer(0, 125000, .01)]
        estimate = history_bandwidth(completed, 1.)
        self.assertAlmostEqual(estimate['estimate_gbps'], .1)
        self.assertEqual(estimate['source_chunk_index'], 0)
        self.assertEqual(estimate['estimate_source'], 'previous_completed_message')

    def test_history_policy_cannot_access_current_or_future_trace_and_rules_match(self):
        from mini_repro.adaptive import next_decision
        chunks = [{'text': {'bytes': 100, 'prefill_seconds': .002}, 'levels': {
            i: {'bytes': i * 200, 'decode_seconds': .001} for i in (1, 2, 3)}}] * 3
        history = (CompletedTransfer(0, 125000, .01),)
        a = history_decision(chunks, 1, history, 1., .03, .01, .001)
        b = next_decision(chunks, 1, .1, .03, .01, 'adaptive', .001)
        for key in ('level', 'remaining_budget_seconds', 'candidate_forecasts_seconds', 'candidate_current_costs_seconds'):
            self.assertEqual(a[key], b[key])
        with self.assertRaises(ValueError):
            history_decision(chunks, 1, history + (CompletedTransfer(1, 1, .1),), 1., .03, .01, .001)

    def test_mode_size_estimator_and_ablation_have_distinct_observation_ids(self):
        c = self.config()
        ids = {ttft_key('t', 'c', 1., 1, 'raw', mode, size, 1) for mode, size in ttft_variants(c, 257)}
        self.assertEqual(len(ids), 3)
        self.assertNotEqual(adaptive_key('a', 'c', 0, 1, 'adaptive', c), adaptive_key('a', 'c', 0, 1, 'adaptive_history', c))
        selected = {k: [{'case_id': k, 'source_id': 0, 'context_tokens': 257}] for k in ('quality', 'controlled', 'triviaqa', 'ttft', 'adaptive')}
        keys = expected_keys(c, selected, jobs(c))
        self.assertEqual(len(keys['ablation']), 12)
        with self.assertRaisesRegex(ValueError, 'Unknown task'):
            expected_keys(c, selected, [{'name': 'bad', 'kind': 'typo'}])
        for section, field, value in (('transport', 'queue_capacity', 2), ('ttft', 'chunk_sizes', [64]),
                                      ('adaptive', 'initial_bandwidth_gbps', 2.), ('ablation', 'repeats', 4)):
            changed = copy.deepcopy(c)
            changed[section][field] = value
            self.assertNotEqual(fingerprint(c), fingerprint(changed))

    def test_bitpacking_4_and_5_bit_symbols_and_non_byte_aligned_tail(self):
        for bits in (4, 5):
            for count in (1, 3, 7, 257):
                values = (np.arange(count) % (1 << bits)).astype(np.uint8)
                packed = pack_bits(values, bits)
                self.assertEqual(len(packed), (count * bits + 7) // 8)
                np.testing.assert_array_equal(unpack_bits(packed, bits, count), values)
        with self.assertRaises(ValueError):
            pack_bits([32], 5)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class MechanismGPUTests(unittest.TestCase):
    @torch.inference_mode()
    def test_entropy_paths_use_same_symbols_scales_and_dequantization_with_257_tail(self):
        from experiments import mechanisms as m
        spec = {'layers': 3, 'key_boundaries': [1, 2, 3], 'value_boundary': 1}
        torch.manual_seed(3)
        blob = torch.randn((3, 2, 2, 257, 16), device='cuda', dtype=torch.float16)
        blob[0, 0] = 0
        blob[1, 1, :, 100, :] *= .00001
        for precision in m.PRECISIONS:
            q = m.quantize(blob, spec, precision)
            from lmcache.storage_backend.serde.cachegen_encoder import torch_quant_vectorized
            groups = blob.permute(1, 0, 3, 2, 4).reshape(6, 257, 32)
            expected_symbols, expected_scales = torch_quant_vectorized(torch.tensor(q.bins, device='cuda', dtype=torch.float32), groups)
            self.assertTrue(torch.equal(q.symbols, expected_symbols))
            self.assertTrue(torch.equal(q.scales, expected_scales))
            reference = m.dequantize(q.symbols, q.scales, q.bins, q.shape)
            restored = []
            for entropy in m.ENTROPY:
                payload, meta = m.serialize(q, entropy)
                symbols, result = m.deserialize(payload)
                self.assertTrue(torch.equal(symbols, q.symbols), (precision, entropy))
                self.assertTrue(torch.equal(result, reference))
                self.assertGreater(len(payload), meta['symbol_data_bytes'])
                self.assertEqual(result.dtype, torch.float16)
                self.assertTrue(torch.isfinite(result).all())
                if entropy == 'arithmetic':
                    self.assertGreater(meta['cdf_tensor_bytes'], 0)
                restored.append(result)
            self.assertTrue(torch.equal(*restored))

    @torch.inference_mode()
    def test_real_gpu_chunked_modes_and_original_whole_text_reference(self):
        import test_shared_framework as shared
        from experiments.ttft import replay, _persist, _load
        engine = shared.SharedEngineGPUTests().make_engine()
        class Tokenizer:
            eos_token_id = 2
            def decode(self, ids, **_):
                return ' '.join(map(str, ids))
            def encode(self, text, **_):
                return list(map(int, text.split()))
        engine.tokenizer = Tokenizer()
        cfg = load_config(REPO / 'experiments/configs/minimal.json', 'tinyllama')
        ids, suffix = [1, 7, 8, 9, 10, 11, 12], [20, 21]
        blob = engine.prefill(ids)
        for method in ('text', 'raw', 'cachegen2'):
            packets = prepare(engine, ids, blob, method, 3)
            self.assertEqual([c.end - c.start for c in packets], [3, 3, 1])
            with tempfile.TemporaryDirectory() as temp:
                _persist(Path(temp), packets, .1)
                reopened, _ = _load(engine, Path(temp))
                self.assertEqual([p.wire for p in reopened], [p.wire for p in packets])
            results = [replay(engine, cfg, packets, None, mode, ids, suffix) for mode in ('chunk_serial', 'chunk_pipeline')]
            self.assertTrue(torch.equal(results[0][0], results[1][0]))
            self.assertEqual(results[0][1]['first_token_id'], results[1][1]['first_token_id'])
            if method == 'raw':
                self.assertTrue(torch.equal(results[0][0], blob))
            if method == 'text':
                self.assertTrue(torch.allclose(results[0][0], blob, atol=.005))
                self.assertEqual(engine.generate(suffix, results[0][0], 3)['generated_ids'], engine.generate(suffix, blob, 3)['generated_ids'])
        whole = prepare(engine, ids, blob, 'text', len(ids))
        expected = engine.first(ids, suffix)
        with patch.object(engine, 'prefill', side_effect=AssertionError('Whole text must keep one combined forward')):
            recovered_ids, row = replay(engine, cfg, whole, None, 'whole', ids, suffix)
        self.assertEqual(recovered_ids, ids)
        self.assertEqual(row['first_token_id'], expected)
