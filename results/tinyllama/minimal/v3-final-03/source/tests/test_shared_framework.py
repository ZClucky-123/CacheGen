"""Regression tests for protocol changes, exact matrices and durable recovery."""
import copy
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from experiments.config import load_config, jobs, METHODS
from experiments.records import Journal, expected_keys, validate_task_records
from experiments.scoring import score_qa, extract_answer
from experiments.datasets import load_triviaqa
from experiments.analysis import cluster_interval, summarize
from mini_repro.experiment_common import pack_uniform, unpack_uniform
from mini_repro.transport import rate_limited_transfer

REPO = Path(__file__).resolve().parents[1]


class SharedProtocolTests(unittest.TestCase):
    def test_both_adapters_use_seven_jobs_and_fixed_scoring(self):
        for name in ('tinyllama', 'mistral'):
            cfg = load_config(REPO / 'experiments/configs/small.json', name)
            self.assertEqual(len(jobs(cfg)), 7)
            self.assertEqual(cfg['model']['name'], name)
            self.assertEqual(set(cfg['quality']['methods']), set(METHODS))

    def test_all_presets_parse_and_close_scales_have_distinct_names(self):
        for path in (REPO / 'experiments/configs').glob('*.json'):
            load_config(path)
        cfg = load_config(REPO / 'experiments/configs/small.json')
        cfg['adaptive']['bandwidth_scales'] = [.01, .010000001]
        self.assertEqual(len({j['name'] for j in jobs(cfg)}), 7)

    def test_first_line_aliases_and_raw_prediction_are_not_rewritten(self):
        raw = '\n\nThe Fruit!\nAdditional words'
        self.assertEqual(score_qa(raw, ['fruit', 'apple']), {'em': 1., 'f1': 1.})
        self.assertEqual(extract_answer(raw), 'The Fruit!')
        self.assertEqual(raw, '\n\nThe Fruit!\nAdditional words')
        self.assertEqual(score_qa('blue flowers\nred flowers', ['red flowers']), {'em': 0., 'f1': .5})
        self.assertEqual(extract_answer('A\u2028B\nC'), 'A\u2028B')

    def test_jsonl_preserves_embedded_unicode_separators(self):
        class Tokenizer:
            def encode(self, text, **kwargs):
                return list(map(ord, text))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'tqa.jsonl'
            prompt = 'question\u2028still the same\u0085record'
            path.write_text(json.dumps({'prompt': prompt, 'answers': ['a']}, ensure_ascii=False) + '\n')
            rows = load_triviaqa(path, Tokenizer(), {'samples': 1, 'max_prompt_tokens': 100})
            self.assertEqual(rows[0]['input_ids'], list(map(ord, prompt)))

    def test_text_wire_retokenizes_and_repairs_multiple_subword_edges(self):
        from experiments.engine import Engine
        class Tokenizer:
            calls = 0
            def decode(self, ids, **kwargs):
                return 'boundary text\u2028preserved'
            def encode(self, text, **kwargs):
                self.calls += 1
                return [10, 2, 3, 40]
        engine = Engine.__new__(Engine)
        engine.tokenizer = Tokenizer()
        payload = engine.text_payload([1, 2, 3, 4])
        self.assertEqual(engine.restore_text(payload), [1, 2, 3, 4])
        self.assertGreaterEqual(engine.tokenizer.calls, 3)
        meta = engine.text_metadata(payload)
        self.assertEqual(meta['patch_inserted_tokens'], 2)
        self.assertEqual(meta['token_patch_count'], 2)
        self.assertGreater(len(payload), meta['text_utf8_bytes'])

    def test_four_bit_odd_nibble_tail_and_subnormal_groups(self):
        blob = torch.tensor([0., .00001, -.00001, .5, -1., 1.], dtype=torch.float16).reshape(1, 2, 1, 3, 1)
        for bits in (4, 8):
            recovered = unpack_uniform(pack_uniform(blob, bits), device='cpu')
            self.assertTrue(torch.isfinite(recovered).all())
            self.assertTrue(torch.allclose(recovered, blob, atol=.001))
        # Three values force a padding nibble; dimensions need not have K/V=2 for pack helper.
        odd = torch.tensor([1., -.5, 0.]).reshape(1, 1, 1, 3, 1)
        self.assertTrue(torch.equal(unpack_uniform(pack_uniform(odd, 4), device='cpu'), odd))

    def test_transport_paces_tail_and_unlimited_payloads_exactly(self):
        payload = bytes(range(256)) * 4096 + b'tail'
        for bw in (None, .1):
            received, seconds = rate_limited_transfer(payload, bw, validate=True)
            self.assertEqual(received, payload)
            if bw:
                self.assertGreaterEqual(seconds, len(payload) * 8 / bw / 1e9 * .95)
        for bw in (0, -1, float('nan')):
            with self.assertRaises(ValueError):
                rate_limited_transfer(payload, bw)

    def test_checkpoint_resume_skips_completed_and_rejects_changed_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'records.sqlite3'
            journal = Journal(path, {'code': 'v2', 'data': 'a'})
            journal.add('q/one/raw', 'q', {'nll': .1})
            journal.close()
            resumed = Journal(path, {'code': 'v2', 'data': 'a'})
            self.assertIn('q/one/raw', resumed.keys)
            with self.assertRaises(ValueError):
                resumed.add('q/one/raw', 'q', {'nll': .2})
            with self.assertRaises(ValueError):
                resumed.validate(['q/one/raw', 'q/two/raw'])
            self.assertEqual(resumed.validate(['q/one/raw']), 1)
            resumed.close()
            with self.assertRaises(ValueError):
                Journal(path, {'code': 'v3', 'data': 'a'})

    def test_corrupted_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'records.sqlite3'
            journal = Journal(path, {})
            journal.add('a', 'q', {'value': 1})
            journal.close()
            db = sqlite3.connect(path)
            db.execute('UPDATE observations SET body=?', ('{"value":2}',))
            db.commit()
            db.close()
            with self.assertRaises(ValueError):
                Journal(path, {})

    def test_complete_matrix_without_offline_prefill_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = Journal(Path(directory) / 'records.sqlite3', {})
            self.addCleanup(journal.close)
            row = {'kind': 'ttft', 'offline_encode_seconds': .1, 'transfer_seconds': .1,
                   'reconstruct_seconds': .1, 'remaining_input_seconds': .1,
                   'ttft_wall_seconds': .3, 'actual_bandwidth_gbps': 1.}
            journal.add('t/one', 't', row)
            with self.assertRaisesRegex(ValueError, 'offline_prefill_seconds'):
                validate_task_records(journal, {'name': 't', 'kind': 'ttft'}, ['t/one'])

    def test_matrix_covers_real_qa_and_all_methods_and_repetitions(self):
        cfg = load_config(REPO / 'experiments/configs/small.json')
        selected = {k: [{'case_id': k, 'source_id': 0, 'context_tokens': 256}] for k in ('quality', 'controlled', 'triviaqa', 'ttft', 'adaptive')}
        expected = expected_keys(cfg, selected, jobs(cfg))
        self.assertEqual(len(expected['quality']), 18)
        self.assertEqual(len(expected['ttft']), 108)
        self.assertEqual(len(expected[jobs(cfg)[2]['name']]), 28)
        self.assertEqual(len(expected['ablation']), 12)

    def test_zero_raw_correct_is_unassessable_and_length_qa_is_separate(self):
        cfg = load_config(REPO / 'experiments/configs/small.json')
        rows = []
        for length in cfg['quality']['lengths']:
            for method in cfg['quality']['methods']:
                rows.append({'kind': 'quality', 'method': method, 'case_id': f'n-{length}', 'source_id': 0,
                    'context_tokens': length, 'nll': 1., 'bytes': 100, 'raw_tensor_bytes': 90,
                    'offline_encode_seconds': 0., 'decode_seconds': 0.})
                rows.append({'kind': 'controlled', 'method': method, 'case_id': f'q-{length}', 'source_id': 0,
                    'context_tokens': length, 'em': float(length == 256), 'f1': float(length == 256)})
        for method in cfg['quality']['methods']:
            rows.append({'kind': 'triviaqa', 'method': method, 'case_id': 'tqa', 'source_id': 0, 'em': 0., 'f1': 0.})
        result = summarize(cfg, rows)
        self.assertTrue(all(r['acceptance'] == 'pass' for r in result['quality'] if r['context_tokens'] == 256))
        self.assertTrue(all(r['acceptance'] == 'unassessable' and r['em_retention'] is None
                            for r in result['quality'] if r['context_tokens'] == 512))

    def test_cluster_intervals_keep_shared_conditions_together(self):
        self.assertIsNone(cluster_interval([('same', 1), ('same', 2)]))
        self.assertEqual(cluster_interval([('a', 1), ('a', 1), ('b', 3), ('b', 3)], draws=200), [1., 3.])

    def test_sealed_archive_includes_logs_and_detects_tampering(self):
        from experiments.artifacts import seal, verify
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / 'run.log').write_text('complete log\n')
            (run / 'config.json').write_text('{}')
            seal(run)
            self.assertEqual(verify(run), 2)
            original = (run / 'artifacts.json').read_bytes()
            (run / 'run.log').write_text('altered')
            with self.assertRaises(ValueError):
                verify(run)
            self.assertEqual((run / 'artifacts.json').read_bytes(), original)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class SharedEngineGPUTests(unittest.TestCase):
    def make_engine(self):
        from transformers import LlamaConfig, LlamaForCausalLM
        from experiments.engine import Engine
        torch.manual_seed(17)
        engine = Engine.__new__(Engine)
        engine.spec = {'layers': 2, 'kv_heads': 2, 'head_dim': 16, 'context_limit': 512,
                       'key_boundaries': [1, 1, 2], 'value_boundary': 1, 'id': 'test/configured-adapter'}
        config = LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, vocab_size=64, max_position_embeddings=512,
            attention_dropout=0., eos_token_id=2)
        engine.model = LlamaForCausalLM(config).half().cuda().eval()
        engine.tokenizer = type('Tokenizer', (), {'eos_token_id': 2, 'decode': lambda self, ids, **kw: str(ids)})()
        return engine

    @torch.inference_mode()
    def test_positions_multitoken_nll_and_cache_mutation(self):
        engine = self.make_engine()
        for ids in ([1, 7, 8, 9], [1, 10, 11, 12, 13, 14, 15]):
            blob = engine.prefill(ids)
            original = blob.clone()
            suffix = [23, 24, 25]
            direct = engine.model(input_ids=torch.tensor([ids + suffix], device='cuda')).logits[0, -1]
            reused, _ = engine.forward(suffix, blob)
            self.assertTrue(torch.allclose(direct, reused[0, -1], atol=.005))
            with patch.object(engine.model.model, 'forward', wraps=engine.model.model.forward) as spy:
                engine.forward(suffix, blob)
                self.assertEqual(spy.call_args.kwargs['position_ids'].tolist(), [list(range(len(ids), len(ids) + 3))])
                self.assertEqual(spy.call_args.kwargs['attention_mask'].shape[1], len(ids) + 3)
            expected = engine.model.generate(torch.tensor([ids + suffix], device='cuda'), max_new_tokens=4, do_sample=False,
                                            pad_token_id=2)[0, len(ids + suffix):].tolist()
            if 2 in expected:
                expected = expected[:expected.index(2)]
            self.assertEqual(engine.generate(suffix, blob, 4)['generated_ids'], expected)
            self.assertTrue(torch.equal(blob, original))
            continuation = [30, 31, 32]
            result = engine.nll(blob, ids, continuation)
            logits = engine.model(input_ids=torch.tensor([ids + continuation[:-1]], device='cuda')).logits
            expected_nll = torch.nn.functional.cross_entropy(logits[0, len(ids)-1:].float(), torch.tensor(continuation, device='cuda'))
            self.assertAlmostEqual(result['nll'], float(expected_nll), places=3)
            self.assertEqual(result['tokens_scored'], 3)

    @torch.inference_mode()
    def test_text_projection_uses_only_last_hidden_position(self):
        engine = self.make_engine()
        shapes = []
        hook = engine.model.lm_head.register_forward_pre_hook(lambda _m, args: shapes.append(args[0].shape))
        try:
            optimized, _ = engine.forward([1, 7, 8, 9])
            full = engine.model(input_ids=torch.tensor([[1, 7, 8, 9]], device='cuda')).logits
        finally:
            hook.remove()
        self.assertEqual([s[1] for s in shapes], [1, 4])
        self.assertTrue(torch.allclose(optimized[0, -1], full[0, -1], atol=.005))

    def test_configured_codec_zero_groups_and_arithmetic_tail(self):
        engine = self.make_engine()
        for length in (1, 257):
            blob = torch.randn(2, 2, 2, length, 16, device='cuda', dtype=torch.float16)
            blob[0, 0] = 0
            for level in (1, 2, 3):
                encoder, decoder = engine.codec(1, level)
                if length > 1:
                    # Reuse an already allocated one-token decoder for a larger payload.
                    decoder.from_bytes(encoder.to_bytes(blob[:, :, :, :1, :]))
                restored = decoder.from_bytes(encoder.to_bytes(blob))
                self.assertEqual(restored.shape, blob.shape)
                self.assertEqual(restored.dtype, blob.dtype)
                self.assertTrue(torch.isfinite(restored).all())
                self.assertTrue(torch.equal(restored[0, 0], blob[0, 0]))
                # Validate every symbol, including the 257th token, against the
                # quantizer itself. Finite/shape checks cannot detect a zero tail.
                groups = blob.permute(0, 1, 3, 2, 4).reshape(2, 2, length, 32)
                bins = torch.stack((encoder.key_bins, encoder.value_bins), dim=1)[:, :, None, None]
                qmax = bins // 2 - 1
                maxima = groups.abs().amax(dim=-1, keepdim=True)
                symbols = torch.round(groups * (qmax / maxima.clamp_min(torch.finfo(blob.dtype).tiny)) + qmax)
                reference = ((symbols - qmax) / qmax * maxima).half()
                actual = restored.permute(0, 1, 3, 2, 4).reshape_as(reference)
                self.assertTrue(torch.allclose(actual, reference, atol=.001, rtol=.001),
                                f'Codec symbol mismatch: length={length}, level={level}')
