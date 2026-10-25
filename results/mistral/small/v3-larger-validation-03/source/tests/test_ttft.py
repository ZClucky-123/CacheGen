import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from mini_repro.ttft import first_token, measurement_schedule, rate_limited_transfer, summarize_ttft


class TTFTTests(unittest.TestCase):
    def test_text_and_restored_kv_build_cache_for_continued_generation(self):
        class Model:
            def __init__(self):
                self.calls = []

            def __call__(self, **kwargs):
                self.calls.append(kwargs)
                logits = torch.zeros(1, kwargs["input_ids"].shape[1], 4)
                logits[0, -1, 2] = 1
                return SimpleNamespace(logits=logits)

        model = Model()
        original_tensor = torch.tensor

        def cpu_tensor(*args, **kwargs):
            kwargs["device"] = "cpu"
            return original_tensor(*args, **kwargs)

        cache = torch.zeros(1, 2, 1, 2, 1)
        with patch("mini_repro.ttft.torch.tensor", side_effect=cpu_tensor):
            text_token = first_token(model, [11, 12], [13])
            cached_token = first_token(model, [11, 12], [13], cache)
        self.assertEqual((text_token, cached_token), (2, 2))
        self.assertEqual(model.calls[0]["input_ids"].tolist(), [[11, 12, 13]])
        self.assertEqual(model.calls[1]["input_ids"].tolist(), [[13]])
        self.assertTrue(all(call["use_cache"] for call in model.calls))

    def test_rate_limited_socket_roundtrips_bytes(self):
        payload = bytes(range(256)) * 100
        received, seconds = rate_limited_transfer(payload, bandwidth_gbps=0.008)
        self.assertEqual(received, payload)
        self.assertGreaterEqual(seconds, len(payload) * 8 / (0.008 * 1e9) * 0.7)
        self.assertLess(seconds, 1.0)

    def test_summary_reports_median_and_p95_by_method_and_bandwidth(self):
        cases = [{"context_tokens": 512, "measurements": [{"method": "cachegen2", "bandwidth_gbps": 1.0,
                                    "ttft_wall_seconds": value}]} for value in (1.0, 2.0, 3.0)]
        row = summarize_ttft(cases)["cachegen2@1Gbps@512tokens"]
        self.assertEqual(row["samples"], 3)
        self.assertEqual(row["median_seconds"], 2.0)
        self.assertAlmostEqual(row["p95_seconds"], 2.9)

    def test_schedule_keeps_warmups_out_of_measured_repetitions(self):
        methods = ["text", "uniform8", "cachegen2", "cachegen3"]
        schedule = measurement_schedule(methods, [0.1, 1.0], warmup=2, repeats=3, seed=9)
        measured = [row for row in schedule if row["phase"] == "measure"]
        warmed = [row for row in schedule if row["phase"] == "warmup"]
        self.assertEqual(len(measured), 24)
        self.assertEqual(len(warmed), 16)
        for bandwidth in (0.1, 1.0):
            for repetition in (1, 2, 3):
                batch = [row for row in measured if row["bandwidth_gbps"] == bandwidth
                         and row["repetition"] == repetition]
                self.assertEqual({row["method"] for row in batch}, set(methods))
                self.assertEqual([row["method_order"] for row in batch], [0, 1, 2, 3])

    def test_schedule_order_is_randomized_and_reproducible(self):
        methods = ["text", "uniform8", "cachegen2", "cachegen3"]
        first = measurement_schedule(methods, [1.0], warmup=1, repeats=5, seed=12)
        self.assertEqual(first, measurement_schedule(methods, [1.0], warmup=1, repeats=5, seed=12))
        self.assertNotEqual(first, measurement_schedule(methods, [1.0], warmup=1, repeats=5, seed=13))
        self.assertEqual(methods, ["text", "uniform8", "cachegen2", "cachegen3"])

    def test_schedule_rejects_invalid_repetition_counts_and_bandwidth(self):
        for override in ({"warmup": -1}, {"repeats": 0}, {"bandwidths": [float("nan")]},
                         {"bandwidths": [0.0]}, {"methods": ["text", "text"]}):
            args = {"methods": ["text"], "bandwidths": [1.0], "warmup": 0,
                    "repeats": 1, "seed": 0}
            args.update(override)
            with self.subTest(override=override), self.assertRaises(ValueError):
                measurement_schedule(**args)

    def test_summary_distinguishes_independent_contexts_from_repeated_timings(self):
        cases = [{"case_id": name, "context_tokens": 512,
                  "measurements": [{"method": "text", "bandwidth_gbps": 1.0,
                                    "ttft_wall_seconds": value} for value in values]}
                 for name, values in (("a", (1.0, 3.0)), ("b", (9.0, 11.0)))]
        row = summarize_ttft(cases)["text@1Gbps@512tokens"]
        self.assertEqual(row["contexts"], 2)
        self.assertEqual(row["timing_repetitions"], 4)
        self.assertEqual(row["repetitions_per_context"], {"a": 2, "b": 2})
        self.assertEqual(row["median_seconds"], 6.0)
        self.assertAlmostEqual(row["p95_seconds"], 10.7)


if __name__ == "__main__":
    unittest.main()
