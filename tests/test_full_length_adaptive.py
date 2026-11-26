"""Run on the server; no Mistral download or model load."""
import json
import math
from pathlib import Path
import unittest

from tools.full_length_adaptive import trace_for, validate, matrix, predict_nll, generate_native, ROOT
from paper_compression.policy import choose


def config():
    return json.loads((ROOT / 'paper_compression/configs/full-length-adaptive.json').read_text())


class FullLengthProtocolTests(unittest.TestCase):
    def test_native_generation_matches_original_and_preserves_prefix(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('CUDA server required')
        from transformers import MistralConfig, MistralForCausalLM
        from experiments.engine import Engine
        from unittest.mock import patch
        class Tokenizer:
            eos_token_id = -1
            def decode(self, ids, **kwargs):
                return ','.join(map(str, ids))
        torch.manual_seed(42)
        engine = Engine.__new__(Engine)
        engine.spec = {'context_limit': 256, 'layers': 2, 'kv_heads': 1, 'head_dim': 16}
        engine.tokenizer = Tokenizer()
        engine.model = MistralForCausalLM(MistralConfig(vocab_size=32, hidden_size=32,
            intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
            num_key_value_heads=1, max_position_embeddings=256, sliding_window=None,
            attn_implementation='eager')).half().cuda().eval()
        blob = engine.prefill([1, 2, 3, 4])
        before = blob.clone()
        expected = engine.generate([5, 6], blob, 5)
        initial = engine.first_state([], [5, 6], blob)
        with patch('experiments.engine.legacy_to_blob', side_effect=AssertionError('Full KV copy')):
            actual = generate_native(engine, [5, 6], blob, 5, initial=initial)
        self.assertEqual(actual, expected)
        self.assertTrue(torch.equal(blob, before))
        engine.tokenizer.eos_token_id = expected['generated_ids'][0]
        self.assertEqual(generate_native(engine, [5, 6], blob, 5), engine.generate([5, 6], blob, 5))

    def test_full_matrix_and_pilot_are_paired(self):
        cfg = validate(config())
        full = matrix(cfg, cfg['source_ids'])
        self.assertEqual(len(full), 1200)
        self.assertEqual(len(set(full)), 1200)
        self.assertEqual(len(matrix(cfg, [0])), 60)
        self.assertEqual({r[2] for r in full}, {.5, 1., 2., 4.})

    def test_trace_covers_actual_chunk_count_and_both_endpoints(self):
        template = [10, 10, 3, 1, .5, 1, 3, 10]
        for length in (8559, 9288):
            chunks = (length + 1535) // 1536
            values = trace_for(template, chunks)
            self.assertEqual(len(values), chunks)
            self.assertEqual(values[0], template[0])
            self.assertEqual(values[-1], template[-1])
            self.assertIn(.5, values)
        self.assertEqual(trace_for([3, 3], 7), [3] * 7)

    def test_invalid_trace_and_config_fail_before_model_load(self):
        for values, count in (([], 4), ([0, 1], 4), ([float('nan')], 4), ([1], 0)):
            with self.assertRaises(ValueError):
                trace_for(values, count)
        cfg = config()
        cfg['repeats'] = 0
        with self.assertRaises(ValueError):
            validate(cfg)

    def test_decisions_use_completed_bandwidth_not_future_trace(self):
        chunk = {'text': {'bytes': 20, 'restore_seconds': 10.},
                 'cachegen3': {'bytes': 1000000, 'restore_seconds': .01},
                 'cachegen2': {'bytes': 500000, 'restore_seconds': .01},
                 'cachegen1': {'bytes': 250000, 'restore_seconds': .01}}
        probe = {'bytes': 125000, 'seconds': .01}
        order = ['cachegen3', 'cachegen2', 'cachegen1']
        fine = choose([chunk], 0, probe, [], .5, 0., .01, .15, .05, order)
        self.assertEqual(fine['method'], 'cachegen3')
        slow = {'bytes': 125000, 'seconds': 1.}
        coarse = choose([chunk], 0, slow, [], 3., 0., .01, .15, .05, order)
        self.assertEqual(coarse['method'], 'cachegen1')
        self.assertEqual(fine['estimate_source'], 'measured_probe')

    def test_answer_nll_empty_targets_needs_no_gpu(self):
        self.assertIsNone(predict_nll(None, [1], None, []))

    def test_fresh_question_teacher_forcing_alignment(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('CUDA server required')
        class FakeEngine:
            def forward(inner, ids, blob, last_only):
                self.assertEqual(ids, [1, 2, 3])  # question [1,2], raw target prefix [3]
                self.assertFalse(last_only)
                logits = torch.zeros((1, 3, 8), device='cuda')
                logits[0, 1, 3] = 9.
                logits[0, 2, 4] = 9.
                return logits, None
        actual = predict_nll(FakeEngine(), [1, 2], None, [3, 4])
        self.assertAlmostEqual(actual, math.log1p(7 * math.exp(-9)), places=6)


if __name__ == '__main__':
    unittest.main()
