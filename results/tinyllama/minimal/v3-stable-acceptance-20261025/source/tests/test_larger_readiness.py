"""Large-context failure modes: skewed entropy, credits and durable identity."""
import tempfile
import copy
import time
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from experiments.artifacts import InputGuard, code_data_identity
from experiments.records import record_memory_peak, validate_timeline
from mini_repro.transport import transfer_chunks


class ReadinessTests(unittest.TestCase):
    def test_resume_keeps_previous_attempt_memory_peak(self):
        manifest = {'attempts': [{'peak_gpu_allocated_bytes': 20, 'peak_gpu_reserved_bytes': 22}, {}]}
        record_memory_peak(manifest, 14, 18)
        self.assertEqual(manifest['peak_gpu_allocated_bytes'], 20)
        self.assertEqual(manifest['peak_gpu_reserved_bytes'], 22)
        self.assertEqual(manifest['attempts'][-1]['peak_gpu_allocated_bytes'], 14)

    def test_input_guard_rejects_edits_and_new_helpers_but_not_same_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'mini_repro').mkdir()
            (root / 'src').mkdir()
            for name in ('env.yaml', 'mini_repro/requirements-mini.txt', 'src/helper.py', 'data.jsonl'):
                (root / name).write_text('original\n')
            cfg = {'data': {'source': 'data.jsonl'}}
            guard = InputGuard(root, cfg, code_data_identity(root, cfg))
            (root / 'src/helper.py').write_text('original\n')
            guard.check(force=True)
            (root / 'src/new.py').write_text('new\n')
            with self.assertRaisesRegex(RuntimeError, 'Code/data changed'):
                guard.check(force=True)
            (root / 'src/new.py').unlink()
            (root / 'data.jsonl').write_text('modified\n')
            with self.assertRaisesRegex(RuntimeError, 'Code/data changed'):
                guard.check(force=True)

    def test_two_queue_credits_do_not_allow_three_messages_ahead(self):
        def restore(index, data):
            time.sleep(.02)
            return data
        result = transfer_chunks([b'x' * 4096] * 6, None, restore,
                                 mode='chunk_pipeline', queue_capacity=2)
        rows = result['chunks']
        for i in range(2, len(rows)):
            self.assertGreaterEqual(rows[i]['send_start'], rows[i - 2]['dequeue_time'])
        self.assertGreater(sum(r['capacity_wait_seconds'] for r in rows), .01)

    def test_timeline_rejects_first_illegal_prefetch_at_capacity_boundary(self):
        row = {'mode': 'chunk_pipeline', 'queue_capacity': 1, 'ttft_wall_seconds': 3.,
               'context_tokens': 2, 'bytes': 2, 'chunks': []}
        for i in range(2):
            row['chunks'].append({'chunk_index': i, 'token_start': i, 'token_end': i+1,
                'message_bytes': 1, 'receive_start': float(i), 'send_start': float(i)+.1,
                'receive_end': float(i)+.4, 'data_available': float(i)+.5,
                'dequeue_time': float(i)+.6, 'restore_start': float(i)+.7, 'restore_end': float(i)+.9})
        validate_timeline(row)
        bad = copy.deepcopy(row)
        bad['chunks'][1].update(receive_start=.1, send_start=.2, receive_end=.3, data_available=.4)
        with self.assertRaisesRegex(ValueError, 'credits'):
            validate_timeline(bad)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class LargeArithmeticGPUTests(unittest.TestCase):
    @torch.inference_mode()
    def test_rare_symbol_with_16_bit_cdf_probability_fits_safe_segment(self):
        import torchac_cuda
        from lmcache.storage_backend.serde.cachegen_encoder import encode_ntokens
        from lmcache.storage_backend.serde.cachegen_decoder import decode_chunk
        from lmcache.storage_backend.serde.cachegen_basics import (
            CACHEGEN_SAFE_ENCODE_TOKENS, CacheGenGPUBytestream)
        n = CACHEGEN_SAFE_ENCODE_TOKENS
        cdf = torch.arange(33, device='cuda', dtype=torch.int32)
        cdf[-1] = 65536
        cdf = cdf.short().reshape(1, 1, 33).expand(2, 32, 33).contiguous()
        symbols = torch.zeros((2, n, 32), device='cuda', dtype=torch.int8)
        buffer = torch.zeros((2, 32, 256), device='cuda', dtype=torch.uint8)
        lengths = torch.zeros((2, 32), device='cuda', dtype=torch.int32)
        data = encode_ntokens(cdf, symbols, buffer, lengths)
        self.assertGreater(int(lengths.max()), n)  # >1 byte/symbol is real.
        self.assertLessEqual(int(lengths.max()), 256)
        restored = torch.empty_like(symbols, dtype=torch.uint8)
        decode_chunk(cdf, CacheGenGPUBytestream(data, lengths, n), restored)
        self.assertTrue(torch.equal(restored, symbols))
        with self.assertRaisesRegex(ValueError, 'safe segments'):
            encode_ntokens(cdf, symbols.repeat(1, 3, 1), buffer, lengths)

    @torch.inference_mode()
    def test_4096_tokens_with_locally_rare_256_token_block_roundtrip_all_symbols(self):
        import test_shared_framework as shared
        from lmcache.storage_backend.serde.cachegen_basics import CacheGenGPUEncoderOutput, CACHEGEN_SAFE_ENCODE_TOKENS
        engine = shared.SharedEngineGPUTests().make_engine()
        encoder, decoder = engine.codec(4096, 3)
        symbols = torch.full((2, 4096, 32), 15, device='cuda', dtype=torch.int8)
        symbols[:, :256] = (torch.arange(256, device='cuda') % 31)[None, :, None].to(torch.int8)
        scales = torch.ones((2, 4096, 1), device='cuda', dtype=torch.float16)
        blob = torch.zeros((2, 2, 2, 4096, 16), device='cuda', dtype=torch.float16)
        with patch('lmcache.storage_backend.serde.cachegen_encoder.torch_quant_vectorized',
                   side_effect=[(symbols, scales), (symbols, scales)]):
            payload = encoder.to_bytes(blob)
        saved = CacheGenGPUEncoderOutput.from_bytes(payload)
        self.assertEqual(sum(c.ntokens for c in saved.data_chunks), 4096)
        self.assertTrue(all(c.ntokens <= CACHEGEN_SAFE_ENCODE_TOKENS for c in saved.data_chunks))
        self.assertTrue(all(int(c.bytestream_lengths.max()) <= 256 for c in saved.data_chunks))
        restored = decoder.from_bytes(payload)
        reference = ((symbols.float() - 15) / 15).half().reshape(2, 4096, 2, 16).permute(0, 2, 1, 3)
        self.assertTrue(torch.equal(restored[:, 0], reference))
        self.assertTrue(torch.equal(restored[:, 1], reference))
