"""Server-only regression suite for the split-locked paper study."""
import json
from pathlib import Path
import unittest
import torch

from paper_compression.config import CodecConfig
from paper_compression.study_config import validate, candidates, family, select_candidates
from paper_compression.quantization import quantize, reconstruct
from paper_compression.probability import make_cdf, ProfileBuilder, GPUProfileBuilder, grouped_histogram
from paper_compression.theory import symbol_bits, estimate
from paper_compression.policy import bandwidth, choose, fixed_strategy_method
from paper_compression.insights import conditional_entropy, cross_entropy

ROOT = Path(__file__).resolve().parents[1]


def configuration():
    return json.loads((ROOT / 'paper_compression/configs/core-study.json').read_text())


class StudyProtocolTests(unittest.TestCase):
    def test_fixed_splits_and_candidate_matrix(self):
        cfg = validate(configuration())
        options = candidates(cfg)
        self.assertEqual(len(options), 50)
        self.assertEqual(sum(family(n) == 'core' for n in options), 20)
        self.assertEqual({c.vector_range for c in options.values()}, {6, 8, 10, 12, 15, 20, 25, 31, 50, 127})
        self.assertEqual(cfg['network']['bandwidth_gbps'], [0.5, 1, 3, 10])
        self.assertEqual(cfg['network']['slo_seconds'], [.5, 1.])
        cfg['splits']['calibration'][0] = 20
        with self.assertRaises(ValueError):
            validate(cfg)

    def test_selection_uses_validation_and_every_length(self):
        cfg = configuration()
        rows, profiles = [], {}
        for name in candidates(cfg):
            profiles[name] = 'profile-' + name
            for source in cfg['splits']['validation']:
                for length in cfg['lengths']:
                    rows.append({'method': name, 'source_id': source, 'context_tokens': length,
                                 'delta_nll': .01, 'estimated_bytes': 100})
        result = select_candidates(cfg, rows, profiles)
        self.assertEqual(result['selection_scope'], 'validation_only')
        for kind, name in result['selected'].items():
            self.assertEqual(family(name), kind)
        options = candidates(cfg)
        self.assertEqual(options[result['selected']['core']].anchor_reference,
                         options[result['selected']['no-layer']].anchor_reference)
        rows[0]['source_id'] = 0
        with self.assertRaises(ValueError):
            select_candidates(cfg, rows, profiles)

    def test_no_feasible_candidate_does_not_silently_relax_gate(self):
        cfg = configuration()
        rows = [{'method': name, 'source_id': s, 'context_tokens': n,
                 'delta_nll': .1, 'estimated_bytes': 1}
                for name in candidates(cfg) for s in cfg['splits']['validation'] for n in cfg['lengths']]
        with self.assertRaises(ValueError):
            select_candidates(cfg, rows, {})


class StudyQuantizationTests(unittest.TestCase):
    def test_original_vs_reconstructed_residual(self):
        blob = torch.tensor([1., .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.1, -.15]).reshape(1, 1, 1, 3, 4).expand(3, 2, 1, 3, 4)
        original = quantize(blob, CodecConfig(layers=3, vector_range=7.5))
        rebuilt = quantize(blob, CodecConfig(layers=3, vector_range=7.5, anchor_reference='reconstructed'))
        self.assertTrue(torch.equal(original.anchor, rebuilt.anchor))
        self.assertFalse(torch.equal(original.delta_scales, rebuilt.delta_scales))
        self.assertEqual(reconstruct(rebuilt).shape, blob.shape)

    def test_no_delta_does_not_add_anchor_to_other_tokens(self):
        blob = torch.arange(12).float().reshape(1, 1, 1, 3, 4).expand(3, 2, 1, 3, 4)
        cfg = CodecConfig(layers=3, use_delta=False, delta_scale_mode='absolute', delta_bin_widths=(1., 1., 1.))
        self.assertTrue(torch.equal(reconstruct(quantize(blob, cfg))[:, :, :, 1:], blob.half()[:, :, :, 1:]))

    def test_theory_uses_shared_probabilities_and_all_overheads(self):
        cdf = make_cdf(torch.zeros((6, 4, 256), dtype=torch.int64))
        symbols = torch.full((6, 11, 4), 255, dtype=torch.uint8)
        self.assertEqual(float(symbol_bits(symbols, cdf)), symbols.numel() * 8.)
        cfg = CodecConfig(layers=3)
        q = quantize(torch.zeros((3, 2, 1, 11, 4)), cfg)
        builder = ProfileBuilder(cfg, {'model': 'test'}, 'dataset')
        builder.add(q, 30, 'input')
        p = builder.finish()
        result = estimate(q, p, metadata_bytes=100)
        self.assertEqual(result['estimated_bytes'], sum(result[k] for k in (
            'cross_entropy_bytes', 'scale_bytes', 'escape_bytes', 'segment_lengths_bytes',
            'termination_allowance_bytes', 'metadata_budget_bytes')))

    def test_grouping_entropy_matches_known_patterns(self):
        symbols = torch.tensor([[[0, 1], [0, 1]], [[1, 0], [1, 0]]], dtype=torch.uint8)
        global_counts = grouped_histogram(symbols, 'global', 2)
        local_counts = grouped_histogram(symbols, 'layer_channel', 2)
        self.assertEqual(conditional_entropy(global_counts), 1.)
        self.assertEqual(conditional_entropy(local_counts), 0.)
        for kind in ('global', 'layer', 'layer_channel', 'token'):
            self.assertEqual(int(grouped_histogram(symbols, kind, 2).sum()), symbols.numel())

    def test_grouping_cross_entropy_uses_calibration_not_validation_fit(self):
        counts = torch.tensor([[10000, 0]], dtype=torch.int64)
        frozen = make_cdf(counts)
        same = torch.tensor([[100, 0]], dtype=torch.int64)
        shifted = torch.tensor([[0, 100]], dtype=torch.int64)
        before = frozen.clone()
        self.assertLess(cross_entropy(same, frozen), cross_entropy(shifted, frozen))
        self.assertTrue(torch.equal(before, frozen))
        self.assertEqual(conditional_entropy(shifted), 0.)
        self.assertGreater(cross_entropy(shifted, frozen), 10.)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA server required')
    def test_gpu_and_reference_calibration_are_equal(self):
        cfg = CodecConfig(layers=3, use_delta=False)
        cpu = ProfileBuilder(cfg, {'model': 'test'}, 'dataset')
        gpu = GPUProfileBuilder(cfg, {'model': 'test'}, 'dataset')
        for source in (30, 31):
            blob = torch.arange(3 * 2 * 2 * 21 * 4).float().reshape(3, 2, 2, 21, 4) / 100
            cpu.add(quantize(blob, cfg), source, 'input')
            gpu.add(quantize(blob.cuda(), cfg), source, 'input')
        a, b = cpu.finish(), gpu.finish()
        self.assertEqual(a.identity, b.identity)


class CausalPolicyTests(unittest.TestCase):
    def test_fixed_selected_and_highest_quality_are_distinct(self):
        self.assertEqual(fixed_strategy_method('fixed', 'range15', 'range127'), 'range15')
        self.assertEqual(fixed_strategy_method('fixed_highest_quality', 'range15', 'range127'), 'range127')
        self.assertEqual(fixed_strategy_method('text', 'range15', 'range127'), 'text')
        self.assertIsNone(fixed_strategy_method('history', 'range15', 'range127'))
        with self.assertRaises(ValueError):
            fixed_strategy_method('unknown', 'range15', 'range127')

    def test_probe_replaces_fixed_initial_bandwidth(self):
        probe = {'bytes': 125000, 'seconds': .01}
        self.assertEqual(bandwidth(probe, [])['estimate_gbps'], .1)
        measured = bandwidth(probe, [{'bytes': 125000, 'seconds': .02}])
        self.assertEqual(measured['estimate_gbps'], .05)
        self.assertEqual(measured['source_chunk_index'], 0)

    def test_margin_and_infeasibility_are_explicit(self):
        costs = [{'text': {'bytes': 100, 'restore_seconds': 2.},
                  'core': {'bytes': 1000000, 'restore_seconds': .01}}]
        probe = {'bytes': 125000, 'seconds': .01}
        decision = choose(costs, 0, probe, [], 1., .95, .1, .15, .05, ['core'])
        self.assertFalse(decision['forecast_feasible'])
        self.assertLess(decision['remaining_budget_seconds'], 0)
        self.assertEqual(decision['estimate_source'], 'measured_probe')
        with self.assertRaises(ValueError):
            choose(costs, 0, probe, [{'bytes': 1, 'seconds': 1}], 1., 0., .1, .15, .05, ['core'])

    def test_recompute_first_then_best_feasible_kv(self):
        probe = {'bytes': 125000, 'seconds': .01}
        costs = [{'text': {'bytes': 100, 'restore_seconds': .1},
                  'fine': {'bytes': 100000, 'restore_seconds': .01},
                  'coarse': {'bytes': 1000, 'restore_seconds': .01}}]
        decision = choose(costs, 0, probe, [], 1., 0., .01, .15, .05, ['fine', 'coarse'])
        self.assertEqual(decision['method'], 'text')
        costs[0]['text']['restore_seconds'] = 10.
        costs[0]['fine']['restore_seconds'] = 10.
        decision = choose(costs, 0, probe, [], 1., 0., .01, .15, .05, ['fine', 'coarse'])
        self.assertEqual(decision['method'], 'coarse')
        self.assertTrue(decision['forecast_feasible'])


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA server required')
class NetworkBridgeTests(unittest.TestCase):
    def test_new_codec_reuses_preparation_and_both_replay_modes(self):
        from paper_compression.codec import PaperCodec
        from paper_compression.network import CodecEngine
        from experiments.chunks import prepare
        from experiments.ttft import replay
        class Model:
            def first(self, context, suffix, blob=None):
                if blob is None or blob.shape[3] != len(context):
                    raise ValueError('Missing or incomplete reconstructed cache')
                return int(blob[0, 0, 0, 0, 0].float() * 100)
        cfg = CodecConfig(layers=3, vector_range=7.5, anchor_reference='reconstructed')
        blob = torch.arange(3 * 2 * 2 * 21 * 4).float().reshape(3, 2, 2, 21, 4).cuda().half() / 100
        builder = GPUProfileBuilder(cfg, {'model': 'synthetic'}, 'dataset')
        builder.add(quantize(blob, cfg), 30, 'input')
        p = builder.finish()
        engine = CodecEngine(Model(), {'core-test': PaperCodec(p, {'model': 'synthetic'}, 'cuda')})
        context, suffix = list(range(21)), [1]
        packets = prepare(engine, context, blob, 'core-test', 10)
        transport = {'transport': {'queue_capacity': 2, 'pacing_quantum_bytes': 65536}}
        serial, a = replay(engine, transport, packets, None, 'chunk_serial', context, suffix)
        pipelined, b = replay(engine, transport, packets, None, 'chunk_pipeline', context, suffix)
        self.assertTrue(torch.equal(serial, pipelined))
        self.assertEqual(a['first_token_id'], b['first_token_id'])
        self.assertTrue(a['byte_identity_checked'] and b['byte_identity_checked'])


if __name__ == '__main__':
    unittest.main()
