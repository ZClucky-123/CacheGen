import unittest

from mini_repro.adaptive import plan_chunks, summarize


class AdaptiveTests(unittest.TestCase):
    def test_adaptive_switches_precision_and_falls_back_to_text(self):
        chunk = {"levels": {1: {"bytes": 50, "decode_seconds": 0.01},
                            2: {"bytes": 100, "decode_seconds": 0.01},
                            3: {"bytes": 200, "decode_seconds": 0.01}},
                 "text": {"bytes": 20, "prefill_seconds": 0.02}}
        chunks = [chunk, chunk, chunk]
        result = plan_chunks(chunks, [0.00008, 0.00002, 0.000004], slo_seconds=0.09, strategy="adaptive")
        self.assertEqual(result["levels"], [3, 1, 0])
        fixed = plan_chunks(chunks, [0.00008] * 3, slo_seconds=0.09, strategy="fixed2")
        self.assertEqual(fixed["levels"], [2, 2, 2])
        self.assertGreater(result["estimated_seconds"], 0)

    def test_summary_counts_violations_and_answer_matches(self):
        rows = [{"measurements": [
            {"strategy": "adaptive", "slo_violation": False, "correct": True, "estimated_ttft_seconds": 0.2},
            {"strategy": "adaptive", "slo_violation": True, "correct": False, "estimated_ttft_seconds": 0.8},
        ]}]
        aggregate = summarize(rows)["adaptive"]
        self.assertEqual((aggregate["runs"], aggregate["slo_violations"], aggregate["correct"]), (2, 1, 1))
        self.assertAlmostEqual(aggregate["mean_estimated_ttft_seconds"], 0.5)


if __name__ == "__main__":
    unittest.main()
