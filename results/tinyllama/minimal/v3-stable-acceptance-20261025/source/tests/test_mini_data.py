import json
import tempfile
import unittest
from pathlib import Path

import importlib


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False, **kwargs):
        return [ord(char) for char in text]


class MiniDataTests(unittest.TestCase):
    def setUp(self):
        try:
            self.data = importlib.import_module("mini_repro.data")
        except ModuleNotFoundError:
            self.fail("mini_repro.data is not implemented")

    def test_longchat_selection_has_requested_lengths_and_source_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "longchat.jsonl"
            rows = [
                {"test_id": 4, "prompt": "abcdefghij", "label": ["x"]},
                {"test_id": 7, "prompt": "klmnopqrst", "label": ["y"]},
            ]
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            cases = self.data.load_longchat_prefixes(path, CharacterTokenizer(), lengths=(4, 6), sample_count=2)
        self.assertEqual(
            [(case["source_id"], case["length"], len(case["input_ids"])) for case in cases],
            [(4, 4, 4), (7, 4, 4), (4, 6, 6), (7, 6, 6)],
        )
        self.assertEqual(cases[0]["input_ids"], [ord(c) for c in "abcd"])

    def test_qa_context_keeps_fact_at_end_with_exact_token_length(self):
        tokens = self.data.build_qa_context(CharacterTokenizer(), "The codeword is cobalt.", target_tokens=100)
        self.assertEqual(len(tokens), 100)
        self.assertEqual(tokens[-len("The codeword is cobalt."):], [ord(c) for c in "The codeword is cobalt."])

    def test_answer_match_respects_word_boundaries(self):
        self.assertTrue(self.data.answer_matches("The answer is cobalt.", "cobalt"))
        self.assertFalse(self.data.answer_matches("The answer is tired.", "red"))
        self.assertFalse(self.data.answer_matches("Wrong answer.\nAnswer: cobalt", "cobalt"))


if __name__ == "__main__":
    unittest.main()
