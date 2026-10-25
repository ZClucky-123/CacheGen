import unittest

import torch

from mini_repro.run import make_codec


@unittest.skipUnless(torch.cuda.is_available(), "CacheGen CUDA codec required")
class CodecLevelsGPUTests(unittest.TestCase):
    def test_all_levels_roundtrip_grouped_query_kv(self):
        torch.manual_seed(4)
        blob = torch.randn((22, 2, 4, 16, 64), device="cuda", dtype=torch.float16)
        sizes = []
        for level in (1, 2, 3):
            encoder, decoder = make_codec(16, level=level)
            payload = encoder.to_bytes(blob)
            restored = decoder.from_bytes(payload)
            self.assertEqual(restored.shape, blob.shape)
            self.assertTrue(torch.isfinite(restored).all())
            sizes.append(len(payload))
        self.assertEqual(len(sizes), 3)


if __name__ == "__main__":
    unittest.main()
