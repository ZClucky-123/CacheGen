import csv
import json
import tempfile
import unittest
from pathlib import Path

from mini_repro.report import generate_report, summarize_quality


def quality_fixture(raw_correct=True):
    return {
        "continuation": [
            {"case_id": "a", "context_tokens": 512, "methods": [
                {"method": "raw", "bytes": 100, "nll": 2.0},
                {"method": "cachegen2", "bytes": 20, "nll": 2.2},
                {"method": "cachegen3", "bytes": 40, "nll": 2.01}]},
            {"case_id": "b", "context_tokens": 512, "methods": [
                {"method": "raw", "bytes": 300, "nll": 4.0},
                {"method": "cachegen2", "bytes": 100, "nll": 4.2},
                {"method": "cachegen3", "bytes": 80, "nll": 4.01}]},
        ],
        "qa": [{"case_id": "qa-a", "context_tokens": 512, "methods": [
            {"method": "raw", "correct": raw_correct},
            {"method": "cachegen2", "correct": False},
            {"method": "cachegen3", "correct": True}]}],
    }


class ReportTests(unittest.TestCase):
    def test_quality_compares_paired_contexts_and_does_not_reward_lossy_size(self):
        rows = {row["method"]: row for row in summarize_quality(quality_fixture())}
        low, high = rows["cachegen2"], rows["cachegen3"]
        self.assertAlmostEqual(low["compression_ratio"], 400 / 120)
        self.assertAlmostEqual(low["mean_nll_delta"], 0.2)
        self.assertFalse(low["meets_descriptive_threshold"])
        self.assertAlmostEqual(high["mean_nll_delta"], 0.01)
        self.assertEqual(high["qa_retention_on_raw_correct"], 1.0)
        self.assertTrue(high["meets_descriptive_threshold"])

    def test_no_raw_correct_questions_or_natural_contexts_cannot_pass_gate(self):
        data = quality_fixture(raw_correct=False)
        self.assertFalse(any(row["meets_descriptive_threshold"]
                             for row in summarize_quality(data)))
        data = quality_fixture()
        data["continuation"] = []
        self.assertFalse(any(row["meets_descriptive_threshold"]
                             for row in summarize_quality(data)))

    def test_quality_threshold_is_per_length_and_missing_qa_counts_as_failure(self):
        data = quality_fixture()
        data["qa"][0]["methods"] = data["qa"][0]["methods"][:1]
        rows = {row["method"]: row for row in summarize_quality(data)}
        self.assertEqual(rows["cachegen3"]["qa_retention_on_raw_correct"], 0.0)
        self.assertFalse(rows["cachegen3"]["meets_descriptive_threshold"])
        data = quality_fixture()
        data["qa"][0]["context_tokens"] = 1024
        rows = summarize_quality(data)
        self.assertFalse(any(row["meets_descriptive_threshold"] for row in rows))

    def test_report_separates_timing_repeats_and_quality_matched_speedups(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixtures = {
                "quality": quality_fixture(),
                "ttft": {"cases": [{"case_id": "a", "context_tokens": 512,
                    "measurements": [
                        {"method": method, "bandwidth_gbps": 1.0,
                         "ttft_wall_seconds": seconds, "realized_bandwidth_gbps": 0.7}
                        for method, values in (("text", (4.0, 6.0)),
                                               ("cachegen2", (1.0, 1.0)),
                                               ("cachegen3", (2.0, 2.0)))
                        for seconds in values]}]},
                "adaptive": {"parameters": {"length": 512, "slo_seconds": 0.5},
                    "cases": [{"case_id": "qa-a", "raw_correct": True,
                               "measurements": [
                        {"trace_id": 0, "repeat": repeat, "strategy": "adaptive",
                         "estimated_ttft_seconds": seconds, "slo_violation": seconds > 0.5,
                         "correct": True}
                        for repeat, seconds in enumerate((0.2, 0.7))]}]},
            }
            jobs = []
            for name, data in fixtures.items():
                filename = name + ".json"
                (root / filename).write_text(json.dumps(data), encoding="utf-8")
                jobs.append({"name": name, "kind": name, "result": filename,
                             "status": "completed", "parameters": {}})
            (root / "manifest.json").write_text(json.dumps({"jobs": jobs,
                "profile": {"purpose": "pipeline-validation-only"}}), encoding="utf-8")
            generated = generate_report(root)
            self.assertTrue(Path(generated["report"]).is_file())
            with (root / "ttft.csv").open() as handle:
                rows = {row["method"]: row for row in csv.DictReader(handle)}
            self.assertEqual(rows["cachegen3"]["contexts"], "1")
            self.assertEqual(rows["cachegen3"]["observations"], "2")
            self.assertEqual(rows["cachegen3"]["quality_comparison"], "meets_descriptive_threshold")
            self.assertEqual(rows["cachegen2"]["quality_comparison"], "unmatched")
            self.assertEqual(rows["cachegen3"]["uniform8_speedup_quality_comparison"], "unmatched")
            self.assertEqual(float(rows["cachegen3"]["speedup_vs_text"]), 2.5)
            self.assertEqual(float(rows["cachegen3"]["mean_effective_bandwidth_gbps"]), 0.7)
            with (root / "adaptive.csv").open() as handle:
                adaptive = next(csv.DictReader(handle))
            self.assertEqual(adaptive["qa_case_trace_pairs"], "1")
            self.assertEqual(adaptive["observations"], "2")
            self.assertEqual(adaptive["slo_violations"], "1")
            report = (root / "report.md").read_text(encoding="utf-8")
            self.assertIn("does not establish statistical noninferiority", report)
            self.assertIn("no live network", report)
            self.assertIn("**Pipeline validation only", report)


if __name__ == "__main__":
    unittest.main()
