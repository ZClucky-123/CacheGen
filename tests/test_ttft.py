import unittest

from mini_repro.ttft import rate_limited_transfer, summarize_ttft


class TTFTTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
