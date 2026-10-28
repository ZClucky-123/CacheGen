"""Run on the server only. These tests do not load Mistral or download data."""
from dataclasses import replace
import io
import tempfile
from pathlib import Path
import unittest

import torch

from paper_compression.config import CodecConfig
from paper_compression.quantization import (
    quantize, reconstruct, split, positions, delta_symbols, restore_delta)
from paper_compression.probability import ProfileBuilder, ProbabilityProfile, make_cdf, validate_cdf
from paper_compression.codec import PaperCodec
from paper_compression import entropy


MODEL = {'model': 'synthetic', 'weight_loading': 'fp16'}


def sample(tokens=21):
    # CPU RNG is independent of available CUDA hardware.
    generator = torch.Generator().manual_seed(tokens)
    return torch.randn((3, 2, 2, tokens, 4), generator=generator).half()


def profile(config=None):
    config = config or CodecConfig(layers=3)
    builder = ProfileBuilder(config, MODEL, 'dataset-test')
    for source in (100, 101):
        builder.add(quantize(sample(source), config), source, f'input-{source}')
    return builder.finish()


class QuantizationTests(unittest.TestCase):
    def test_config_boundaries_and_disclosure(self):
        cfg = CodecConfig(layers=32)
        self.assertEqual(cfg.boundaries, (10, 21, 32))
        self.assertEqual(cfg.widths(), [.5] * 10 + [1.] * 11 + [1.5] * 11)
        self.assertEqual(CodecConfig.from_dict(cfg.to_dict()), cfg)
        self.assertNotEqual(replace(cfg, delta_scale_mode='absolute').identity, cfg.identity)
        with self.assertRaises(ValueError):
            replace(cfg, parameter_status='verified')

    def test_same_anchor_not_previous_token(self):
        cfg = CodecConfig(layers=3, delta_scale_mode='absolute')
        blob = torch.arange(21).float().view(1, 1, 1, 21, 1).expand(3, 2, 1, 21, 1)
        anchor, delta = split(blob, cfg)
        self.assertEqual(anchor[0, :, 0].tolist(), [0., 10., 20.])
        self.assertEqual(delta[0, :, 0].tolist(), list(map(float, range(1, 10))) * 2)
        # Width .5 and one channel make both anchor and delta reconstruction exact.
        q = quantize(blob, cfg)
        self.assertTrue(torch.equal(reconstruct(q)[0], blob.half()[0]))

    def test_tail_lengths_and_zero_vectors(self):
        cfg = CodecConfig(layers=3)
        for tokens in (1, 9, 10, 11, 257, 4096):
            with self.subTest(tokens=tokens):
                blob = torch.zeros((3, 2, 2, tokens, 4), dtype=torch.float16)
                q = quantize(blob, cfg)
                anchors, deltas, parents = positions(tokens, 10, 'cpu')
                self.assertEqual(len(anchors), (tokens + 9) // 10)
                self.assertEqual(len(deltas) + len(anchors), tokens)
                self.assertTrue(torch.equal(anchors[parents], (deltas // 10) * 10))
                self.assertTrue(torch.equal(reconstruct(q), blob))

    def test_escape_roundtrip_does_not_clip(self):
        cfg = CodecConfig(layers=3)
        values = torch.tensor([-10000, -128, -127, 0, 127, 128, 10000], dtype=torch.int32)
        symbols, escaped = delta_symbols(values, cfg)
        self.assertEqual(symbols.tolist(), [255, 255, 0, 127, 254, 255, 255])
        self.assertTrue(torch.equal(restore_delta(symbols, escaped, cfg), values))
        with self.assertRaises(ValueError):
            restore_delta(symbols, escaped[:-1], cfg)

    def test_corrupt_reconstruction_rejected(self):
        q = quantize(sample(), CodecConfig(layers=3))
        q.anchor[0, 0, 0] = 255
        with self.assertRaises(ValueError):
            reconstruct(q)
        q = quantize(sample(), CodecConfig(layers=3))
        q.anchor_scales[0, 0, 0] = float('nan')
        with self.assertRaises(ValueError):
            reconstruct(q)


class ProbabilityTests(unittest.TestCase):
    def test_positive_cdf_including_unseen_symbols(self):
        counts = torch.zeros((6, 8, 256), dtype=torch.int64)
        counts[..., 0] = 1000000
        cdf = make_cdf(counts)
        validate_cdf(cdf, 6, 8, 256)
        self.assertTrue(bool((cdf[..., -1] == 0).all()))
        cdf[..., 1] = 0
        with self.assertRaises(ValueError):
            validate_cdf(cdf, 6, 8, 256)

    def test_model_and_calibration_isolation(self):
        p = profile()
        p.require_model(MODEL)
        p.require_held_out(0, 'dataset-test')
        for source, dataset in ((100, 'dataset-test'), (0, 'other-dataset')):
            with self.assertRaises(ValueError):
                p.require_held_out(source, dataset)
        with self.assertRaises(ValueError):
            p.require_model({'model': 'other'})
        builder = ProfileBuilder(p.config, MODEL, 'dataset-test')
        q = quantize(sample(), p.config)
        builder.add(q, 1, 'same')
        with self.assertRaises(ValueError):
            builder.add(q, 1, 'same')

    def test_profile_immutable_file_and_fingerprint(self):
        p = profile()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'profile.pt'
            p.save(path)
            self.assertEqual(ProbabilityProfile.load(path).identity, p.identity)
            with self.assertRaises(FileExistsError):
                p.save(path)
            obj = torch.load(path, weights_only=True)
            obj['metadata']['calibration_inputs']['100'] = 'tampered'
            torch.save(obj, path)
            with self.assertRaises(ValueError):
                ProbabilityProfile.load(path)


class ArithmeticTests(unittest.TestCase):
    def assert_stream(self, backend, tokens, rare=False, channels=32):
        counts = torch.zeros((2, channels, 256), dtype=torch.int64)
        if rare:
            counts[..., 0] = 1000000
            symbols = torch.full((2, tokens, channels), 255, dtype=torch.uint8)
        else:
            symbols = torch.arange(2 * tokens * channels).remainder(256).to(torch.uint8).reshape(2, tokens, channels)
        cdf = make_cdf(counts)
        packet = entropy.encode(symbols, cdf, backend)
        restored = entropy.decode(packet, cdf, tuple(symbols.shape), 'cpu')
        self.assertTrue(torch.equal(symbols, restored))
        return packet, cdf, symbols

    def test_cpu_full_alphabet_and_empty_tail(self):
        for tokens in (0, 1, 257):
            with self.subTest(tokens=tokens):
                self.assert_stream('cpu', tokens)

    def test_cpu_rare_symbols_and_corrupt_segment(self):
        packet, cdf, symbols = self.assert_stream('cpu', 96, rare=True)
        packet['segments'][0]['token_start'] += 1
        with self.assertRaises(ValueError):
            entropy.decode(packet, cdf, tuple(symbols.shape), 'cpu')

    def check_codec(self, backend, tokens=257):
        p = profile()
        codec = PaperCodec(p, MODEL, backend)
        q = quantize(sample(tokens), p.config)
        payload, stats = codec.encode_quantized(q)
        restored = codec.decode_quantized(payload, 'cpu')
        self.assertEqual(stats['complete_bytes'], len(payload))
        self.assertEqual(stats['profile_bytes_in_packet'], 0)
        self.assertTrue(torch.equal(q.anchor, restored.anchor))
        self.assertTrue(torch.equal(q.delta, restored.delta))
        self.assertTrue(torch.equal(q.anchor_scales, restored.anchor_scales))
        self.assertTrue(torch.equal(q.delta_scales, restored.delta_scales))
        self.assertTrue(torch.equal(reconstruct(q), reconstruct(restored)))
        obj = torch.load(io.BytesIO(payload), weights_only=True)
        self.assertNotIn('anchor_cdf', obj)
        obj['profile_identity'] = 'wrong'
        stream = io.BytesIO()
        torch.save(obj, stream)
        with self.assertRaises(ValueError):
            codec.decode_quantized(stream.getvalue(), 'cpu')
        return codec, q

    def test_cpu_codec_and_anchor_only_packet(self):
        self.check_codec('cpu')
        self.check_codec('cpu', tokens=1)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA server required')
    def test_cuda_all_byte_values_tails_and_worst_probability(self):
        for tokens in (1, 257, 4096):
            with self.subTest(tokens=tokens):
                self.assert_stream('cuda', tokens)
        self.assert_stream('cuda', 96, rare=True)
        self.assert_stream('cuda', 257, channels=5)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA server required')
    def test_cpu_cuda_quantized_identity(self):
        cuda_codec, q = self.check_codec('cuda')
        cpu_codec = PaperCodec(cuda_codec.profile, MODEL, 'cpu')
        # Each container declares its backend; either codec can restore either packet.
        for sender, receiver in ((cpu_codec, cuda_codec), (cuda_codec, cpu_codec)):
            payload, _ = sender.encode_quantized(q)
            self.assertTrue(torch.equal(receiver.decode(payload, 'cpu'), reconstruct(q)))


if __name__ == '__main__':
    unittest.main()
