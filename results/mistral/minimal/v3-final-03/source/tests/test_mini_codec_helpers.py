import io
import unittest

import torch

from mini_repro.run import blob_to_legacy, serialized_size, uniform8_roundtrip


class CodecHelperTests(unittest.TestCase):
    def test_uniform8_keeps_shape_and_counts_metadata(self):
        blob = torch.tensor([[[[[1.0, -0.5], [0.0, 0.25]]], [[[0.0, 0.0], [0.5, -1.0]]]]])
        # [layers=1, K/V=2, heads=1, tokens=2, dim=2]
        restored, size = uniform8_roundtrip(blob)
        self.assertEqual(restored.shape, blob.shape)
        self.assertTrue(torch.isfinite(restored).all())
        self.assertLess((restored - blob).abs().max().item(), 0.01)
        self.assertGreater(size, blob.numel())

    def test_blob_to_legacy_keeps_layer_order(self):
        blob = torch.arange(24).reshape(2, 2, 1, 3, 2)
        cache = blob_to_legacy(blob)
        self.assertEqual(len(cache), 2)
        self.assertTrue(torch.equal(cache[1][0], blob[1, 0].unsqueeze(0)))

    def test_serialized_size_uses_torch_container(self):
        tensor = torch.zeros((2, 3), dtype=torch.float16)
        self.assertGreater(serialized_size(tensor), tensor.numel() * tensor.element_size())


if __name__ == "__main__":
    unittest.main()
