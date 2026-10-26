import unittest
from unittest.mock import patch
import torch
from experiments.numerics import compare_logits, compare_kv, audit_generation


class NumericalGenerationTests(unittest.TestCase):
    def test_one_ulp_flip_is_recorded_not_claimed_identical(self):
        ref = torch.tensor([20., 19.984375, -2.])
        other = torch.tensor([19.984375, 20., -2.])
        row = compare_logits(ref, other)
        self.assertTrue(row['passed'])
        self.assertFalse(row['argmax_identical'])
        self.assertTrue(row['fp16_near_tie'])
        self.assertEqual(row['fp16_ulp'], .015625)

    def test_clear_margin_flip_is_rejected_even_with_small_absolute_error(self):
        row = compare_logits(torch.tensor([1., .99]), torch.tensor([.99, 1.]))
        self.assertFalse(row['passed'])
        self.assertFalse(row['fp16_near_tie'])

    def test_same_argmax_does_not_hide_corrupted_remaining_logits(self):
        row = compare_logits(torch.tensor([10., 0., -2.]), torch.tensor([10., 0., -1.]))
        self.assertTrue(row['argmax_identical'])
        self.assertFalse(row['numerically_close'])
        self.assertFalse(row['passed'])

    def test_nonfinite_logits_are_rejected(self):
        self.assertFalse(compare_logits(torch.tensor([1., 0.]), torch.tensor([float('nan'), 0.]))['passed'])

    def fake_engine(self):
        class Engine:
            tokenizer = type('Tokenizer', (), {'eos_token_id': 99})()
            calls = 0
            def forward(self, ids, blob):
                self.calls += 1
                scores = [20., 19.984375] if self.calls == 1 else [19.984375, 20.]
                return torch.tensor([[scores]]), None
        return Engine()

    def test_later_near_tie_cannot_hide_an_unexplained_first_divergence(self):
        with patch('experiments.numerics.legacy_to_blob', side_effect=lambda x: x):
            with self.assertRaisesRegex(RuntimeError, 'First generation divergence'):
                audit_generation(self.fake_engine(), [1], None, 2, [0, 0],
                    [torch.tensor([20., 19.984375])] * 2, actual_ids=[1, 0])

    def test_audited_first_divergence_records_real_outputs(self):
        with patch('experiments.numerics.legacy_to_blob', side_effect=lambda x: x):
            row = audit_generation(self.fake_engine(), [1], None, 2, [0, 0],
                [torch.tensor([20., 19.984375])] * 2, actual_ids=[0, 1])
        self.assertEqual(row['first_divergence_step'], 1)
        self.assertFalse(row['generated_ids_identical'])
        self.assertEqual(row['candidate_generated_ids'], [0, 1])

    def test_kv_vector_check_records_near_zero_component_outlier(self):
        ref = torch.ones((1, 2, 1, 2, 128), dtype=torch.float16) * 2
        ref[0, 0, 0, 0, 0] = .2
        other = ref.clone()
        other[0, 0, 0, 0, 0] += .12
        row = compare_kv(ref, other)
        self.assertFalse(row['elementwise_allclose'])
        self.assertGreater(row['elementwise_outlier_count'], 0)
        self.assertTrue(row['rms_within_bound'])
        self.assertFalse(row['passed'])

    def test_kv_vector_check_rejects_bad_token_hidden_by_global_average(self):
        ref = torch.zeros((1, 2, 1, 1024, 128), dtype=torch.float16)
        other = ref.clone()
        other[0, 0, 0, 500] = .5
        row = compare_kv(ref, other)
        self.assertLess(row['mean_abs_error'], .001)
        self.assertFalse(row['passed'])

    def test_kv_absolute_tolerance_keeps_units_across_head_sizes(self):
        for dim in (16, 64, 128):
            ref = torch.zeros((1, 2, 1, 2, dim), dtype=torch.float16)
            self.assertTrue(compare_kv(ref, ref + .05)['passed'])
            self.assertFalse(compare_kv(ref, ref + .15)['passed'])

    def test_kv_position_or_head_corruption_is_rejected(self):
        torch.manual_seed(5)
        ref = torch.randn((2, 2, 2, 32, 16), dtype=torch.float16)
        self.assertFalse(compare_kv(ref, ref.flip(3))['passed'])

    def test_small_global_rms_cannot_hide_pointwise_outlier(self):
        reference = torch.zeros(256)
        reference[0] = 20.
        candidate = reference.clone()
        candidate[100] = .2
        row = compare_logits(reference, candidate)
        self.assertFalse(row['elementwise_allclose'])
        self.assertFalse(row['passed'])
        self.assertTrue(row['argmax_identical'])

    def test_global_rms_cannot_average_away_wrong_top_logit(self):
        reference = torch.zeros(65536)
        reference[0] = 20.
        candidate = reference.clone()
        candidate[100] = 21.
        row = compare_logits(reference, candidate)
        self.assertLess(row['rms_logit_error'], .1)
        self.assertFalse(row['competitive_logits_close'])
        self.assertFalse(row['passed'])

    def test_fp16_zero_and_subnormal_ulp_is_not_overestimated(self):
        row = compare_logits(torch.tensor([0., -.000001]), torch.tensor([-.000001, 0.]))
        self.assertEqual(row['fp16_ulp'], 2. ** -24)
        self.assertFalse(row['fp16_near_tie'])
        self.assertFalse(row['passed'])
