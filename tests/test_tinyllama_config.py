import unittest

from lmcache.storage_backend.serde.cachegen_basics import CacheGenConfig


class TinyLlamaConfigTests(unittest.TestCase):
    def test_tinyllama_has_22_layer_codec_schedule(self):
        config = CacheGenConfig.from_model_name("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        self.assertEqual(
            (config.key_first_layers, config.key_second_layers, config.key_third_layers),
            (7, 14, 22),
        )
        self.assertEqual(
            (config.key_first_bins, config.key_second_bins, config.key_third_bins),
            (32, 16, 16),
        )
        self.assertEqual(
            (config.value_first_layers, config.value_first_bins, config.value_second_bins),
            (2, 32, 16),
        )


if __name__ == "__main__":
    unittest.main()