import unittest

from mini_repro.adaptive import plan_chunks, summarize


class AdaptiveTests(unittest.TestCase):
    def test_adaptive_switches_precision_and_falls_back_to_text(self):
        chunk = {"levels": {1: {"bytes": 50, "decode_seconds": 0.01},
                            2: {"bytes": 100, "decode_seconds": 0.01},
                            3: {"bytes": 200, "decode_seconds": 0.01}},
                 "text": {"bytes": 20, "prefill_seconds": 0.02}}
        chunks = [chunk, chunk, chunk]
        result = plan_chunks(chunks, [0.00008, 0.00002, 0.000004], slo_seconds=0.09, strategy="adaptive_static")
        self.assertEqual(result["levels"], [3, 1, 0])
        fixed = plan_chunks(chunks, [0.00008] * 3, slo_seconds=0.09, strategy="fixed2")
        self.assertEqual(fixed["levels"], [2, 2, 2])
        self.assertGreater(result["estimated_seconds"], 0)

    def test_remaining_budget_reuses_time_saved_by_small_chunk(self):
        def chunk(sizes):
            return {"levels": {level: {"bytes": size, "decode_seconds": 0.0}
                               for level, size in enumerate(sizes, 1)},
                    "text": {"bytes": 1, "prefill_seconds": 0.1}}
        chunks = [chunk((5, 8, 10)), chunk((40, 60, 80))]
        remaining = plan_chunks(chunks, [0.000008] * 2, 0.1)
        static = plan_chunks(chunks, [0.000008] * 2, 0.1, "adaptive_static")
        self.assertEqual(remaining["levels"], [3, 3])
        self.assertEqual(static["levels"], [3, 1])
        self.assertTrue(remaining["feasible"])

    def test_first_token_reserve_changes_selected_precision(self):
        chunk = {"levels": {level: {"bytes": size, "decode_seconds": 0.01}
                            for level, size in ((1, 40), (2, 60), (3, 80))},
                 "text": {"bytes": 1, "prefill_seconds": 0.2}}
        without_reserve = plan_chunks([chunk], [0.000008], 0.1)
        with_reserve = plan_chunks([chunk], [0.000008], 0.1, first_token_reserve=0.03)
        self.assertEqual(without_reserve["levels"], [3])
        self.assertEqual(with_reserve["levels"], [2])
        self.assertTrue(with_reserve["feasible"])

    def test_current_selection_does_not_read_future_bandwidth(self):
        chunk = {"levels": {level: {"bytes": size, "decode_seconds": 0.01}
                            for level, size in ((1, 10), (2, 20), (3, 30))},
                 "text": {"bytes": 1, "prefill_seconds": 0.2}}
        fast_future = plan_chunks([chunk, chunk], [0.000008, 0.00008], 0.09)
        slow_future = plan_chunks([chunk, chunk], [0.000008, 0.0000008], 0.09)
        self.assertEqual(fast_future["levels"][0], slow_future["levels"][0])

    def test_unavoidable_deadline_miss_is_explicit(self):
        chunk = {"levels": {level: {"bytes": 100 * level, "decode_seconds": 0.1}
                            for level in (1, 2, 3)},
                 "text": {"bytes": 1, "prefill_seconds": 0.2}}
        result = plan_chunks([chunk], [0.000008], 0.01)
        self.assertFalse(result["feasible"])
        self.assertFalse(result["decisions"][0]["forecast_feasible"])

    def test_summary_does_not_count_timing_repeats_as_qa_samples(self):
        rows = [{"measurements": [
            {"trace_id": 0, "strategy": "adaptive", "repeat": 0,
             "slo_violation": False, "correct": True, "estimated_ttft_seconds": 0.2},
            {"trace_id": 0, "strategy": "adaptive", "repeat": 1,
             "slo_violation": False, "correct": True, "estimated_ttft_seconds": 0.3},
        ]}]
        aggregate = summarize(rows)["adaptive"]
        self.assertEqual(aggregate["runs"], 2)
        self.assertEqual(aggregate["qa_cases"], 1)
        self.assertEqual(aggregate["qa_correct"], 1)

    def test_summary_distinguishes_unique_facts_from_trace_conditions(self):
        rows = [{"case_id": "one-fact", "measurements": [
            {"trace_id": trace, "strategy": "adaptive", "repeat": repeat,
             "slo_violation": False, "correct": True, "estimated_ttft_seconds": 0.2}
            for trace in (0, 1) for repeat in (0, 1)
        ]}]
        aggregate = summarize(rows)["adaptive"]
        self.assertEqual(aggregate["runs"], 4)
        self.assertEqual(aggregate["unique_fact_cases"], 1)
        self.assertEqual(aggregate["qa_conditions"], 2)
        self.assertEqual(aggregate["qa_cases"], aggregate["qa_conditions"])

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
