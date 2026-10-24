import json
import tempfile
import unittest
from pathlib import Path

import torch

from mini_repro.experiment_common import (
    build_positioned_case,
    choose_strategy,
    load_longchat_segments,
    pack_uniform,
    unpack_uniform,
)


class CharTokenizer:
    def encode(self, text, add_special_tokens=False, **kwargs):
        return [ord(c) for c in text]


class ExperimentCommonTests(unittest.TestCase):
    def test_packed_four_bit_roundtrip_and_size(self):
        blob = torch.tensor([[[[[1., -0.5], [0., 0.25]]], [[[0., 0.], [0.5, -1.]]]]], dtype=torch.float16)
        payload = pack_uniform(blob, bits=4)
        recovered = unpack_uniform(payload, device="cpu")
        self.assertEqual(recovered.shape, blob.shape)
        self.assertTrue(torch.isfinite(recovered).all())
        self.assertLess((recovered - blob).abs().float().max().item(), 0.15)
        larger = torch.linspace(-1, 1, 512, dtype=torch.float16).reshape(1, 2, 2, 64, 2)
        self.assertLess(len(pack_uniform(larger, bits=4)), len(pack_uniform(larger, bits=8)))

    def test_positioned_fact_changes_location_without_leaking_answer_to_filler(self):
        tokenizer = CharTokenizer()
        cases = [build_positioned_case(tokenizer, 0, 200, position) for position in ("start", "middle", "end")]
        self.assertTrue(all(len(case["input_ids"]) == 200 for case in cases))
        self.assertTrue(all(case["answer"] == cases[0]["answer"] for case in cases))
        starts = [case["fact_start"] for case in cases]
        self.assertEqual(starts, sorted(starts))
        answer = cases[0]["answer"]
        for case in cases:
            context = "".join(map(chr, case["input_ids"]))
            self.assertEqual(context.count(answer), 1)

    def test_longchat_segments_include_unseen_continuation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text(json.dumps({"test_id": 7, "prompt": "abcdefghijklmnop"}) + "\n", encoding="utf-8")
            segments = load_longchat_segments(path, CharTokenizer(), lengths=(8,), sample_count=1, continuation_tokens=4)
        self.assertEqual(segments[0]["context_ids"], list(map(ord, "abcdefgh")))
        self.assertEqual(segments[0]["continuation_ids"], list(map(ord, "ijkl")))

    def test_deadline_selector_prefers_highest_precision_that_fits(self):
        choices = {3: {"bytes": 200, "decode_seconds": 0.01}, 2: {"bytes": 100, "decode_seconds": 0.01}, 1: {"bytes": 50, "decode_seconds": 0.01}}
        self.assertEqual(choose_strategy(choices, bandwidth_gbps=0.000008, budget_seconds=0.12), 2)
        self.assertEqual(choose_strategy(choices, bandwidth_gbps=0.000008, budget_seconds=0.02), 0)


if __name__ == "__main__":
    unittest.main()
