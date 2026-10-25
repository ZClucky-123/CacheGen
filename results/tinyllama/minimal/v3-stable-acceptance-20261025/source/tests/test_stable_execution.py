import contextlib
import unittest
from pathlib import Path
import torch
from experiments.config import load_config, validate_config
from experiments.stable import TokenTiledLinear, explicit_causal_mask, configure_execution, install_stable_transformer

REPO = Path(__file__).resolve().parents[1]
STABLE = {'linear_tile_tokens': 64, 'explicit_causal_mask': True,
          'sdpa_backend': 'efficient', 'fp16_reduced_precision_reduction': False}


@contextlib.contextmanager
def configured():
    old = (torch.backends.cuda.flash_sdp_enabled(), torch.backends.cuda.mem_efficient_sdp_enabled(),
           torch.backends.cuda.math_sdp_enabled(), torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)
    configure_execution(STABLE)
    try:
        yield
    finally:
        torch.backends.cuda.enable_flash_sdp(old[0])
        torch.backends.cuda.enable_mem_efficient_sdp(old[1])
        torch.backends.cuda.enable_math_sdp(old[2])
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = old[3]


class StableExecutionTests(unittest.TestCase):
    def test_linear_keeps_row_values_and_tail_under_chunking(self):
        torch.manual_seed(3)
        base = torch.nn.Linear(16, 32)
        layer = TokenTiledLinear(base, 64)
        inputs = torch.randn(1, 129, 16)
        full = layer(inputs)
        chunks = torch.cat([layer(inputs[:, i:i+64]) for i in range(0, 129, 64)], dim=1)
        self.assertTrue(torch.equal(full, chunks))
        self.assertTrue(torch.allclose(full, base(inputs), atol=1e-6))
        self.assertIs(layer.weight, base.weight)

    def test_explicit_mask_uses_absolute_cache_positions_and_padding(self):
        module = type('Module', (), {'embed_tokens': torch.nn.Embedding(32, 8).half()})()
        cache = type('Cache', (), {'get_seq_length': lambda self: 3})()
        args, kwargs = explicit_causal_mask(module, (), {'input_ids': torch.tensor([[10, 11]]),
            'past_key_values': cache, 'cache_position': torch.tensor([3, 4]),
            'attention_mask': torch.tensor([[1, 0, 1, 1, 1]])})
        mask = kwargs['attention_mask']
        self.assertEqual(mask.shape, (1, 1, 2, 5))
        self.assertEqual(mask[0, 0, 0].eq(0).tolist(), [True, False, True, True, False])
        self.assertEqual(mask[0, 0, 1].eq(0).tolist(), [True, False, True, True, True])

    def test_config_rejects_unaligned_stable_chunks(self):
        cfg = load_config(REPO / 'experiments/configs/larger-mistral.json')
        self.assertEqual(cfg['execution']['linear_tile_tokens'], 256)
        cfg['ttft']['chunk_sizes'] = [128]
        with self.assertRaisesRegex(ValueError, 'align'):
            validate_config(cfg)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class StableExecutionGPUTests(unittest.TestCase):
    @torch.inference_mode()
    def test_gqa_prefill_positions_tail_generation_and_last_only_head(self):
        import test_shared_framework as shared
        engine = shared.SharedEngineGPUTests().make_engine()
        with configured():
            install_stable_transformer(engine.model, 64)
            self.assertFalse(isinstance(engine.model.lm_head, TokenTiledLinear))
            self.assertFalse(any(m.training for m in engine.model.modules()))
            for length in (128, 129):
                ids = [1] + [7 + i % 40 for i in range(length - 1)]
                full, chunks = engine.prefill(ids), None
                for i in range(0, length, 64):
                    chunks = engine.prefill(ids[i:i+64], chunks)
                self.assertTrue(torch.allclose(full, chunks, atol=.005))
                if length == 128:
                    self.assertTrue(torch.equal(full, chunks))
                self.assertEqual(engine.generate([25, 26], full, 4)['generated_ids'],
                                 engine.generate([25, 26], chunks, 4)['generated_ids'])
                projected = []
                hook = engine.model.lm_head.register_forward_pre_hook(lambda _, args: projected.append(args[0].shape[1]))
                try:
                    engine.first([], [25, 26], chunks)
                finally:
                    hook.remove()
                self.assertEqual(projected, [1])

    @torch.inference_mode()
    def test_original_fastchat_quantized_weights_are_reused_without_recompression(self):
        from fastchat.model.compression import CLinear
        torch.manual_seed(9)
        base = CLinear(torch.randn((128, 64), device='cuda', dtype=torch.float16))
        layer = TokenTiledLinear(base, 64)
        inputs = torch.randn((1, 129, 64), device='cuda', dtype=torch.float16)
        with configured():
            full = layer(inputs)
            chunks = torch.cat([layer(inputs[:, i:i+64]) for i in range(0, 129, 64)], dim=1)
        self.assertIs(layer.weight, base.weight)
        self.assertTrue(torch.equal(full, chunks))
        self.assertEqual(full.dtype, torch.float16)
