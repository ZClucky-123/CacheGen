"""Small, deterministic inputs for the local CacheGen check."""

import json
import re
from pathlib import Path


QA_CASES = (
    ("museum", "In the museum log, the blue vase was donated by Clara.", "Who donated the blue vase?", "Clara"),
    ("observatory", "In the observatory log, the telescope codeword is cobalt.", "What is the telescope codeword?", "cobalt"),
    ("library", "In the library record, shelf 17 contains the atlas of rivers.", "Which shelf contains the atlas of rivers?", "17"),
    ("garden", "In the garden notes, the winter flowers were planted by Maya.", "Who planted the winter flowers?", "Maya"),
    ("station", "In the station report, the northbound train departs from platform six.", "Which platform has the northbound train?", "six"),
)

FILLER = (
    "The archive contains routine notes about weather, maintenance, and daily schedules. "
    "These notes do not change the final fact in the record. "
)


def load_longchat_prefixes(path, tokenizer, lengths=(512, 1024), sample_count=3):
    """Return token prefixes from the first sample_count bundled LongChat rows."""
    if sample_count <= 0 or any(length <= 0 for length in lengths):
        raise ValueError("sample count and lengths must be positive")
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) == sample_count:
                break
    if len(rows) < sample_count:
        raise ValueError("not enough LongChat rows")
    # The source prompts are much longer than TinyLlama's window. Only their
    # prefixes are fed to the model; suppress the tokenizer's length warning.
    encoded = [
        (row["test_id"], tokenizer.encode(row["prompt"], add_special_tokens=False, verbose=False))
        for row in rows
    ]
    cases = []
    for length in lengths:
        for source_id, tokens in encoded:
            if len(tokens) < length:
                raise ValueError(f"LongChat row {source_id} has fewer than {length} tokens")
            cases.append({
                "case_id": f"longchat-{source_id}-{length}",
                "source_id": source_id,
                "length": length,
                "input_ids": tokens[:length],
            })
    return cases


def build_qa_context(tokenizer, fact, target_tokens=512):
    """Place the answer-bearing fact at the end of a fixed-length context."""
    fact_ids = tokenizer.encode(fact, add_special_tokens=False)
    filler_ids = tokenizer.encode(FILLER, add_special_tokens=False)
    if target_tokens < len(fact_ids) or not filler_ids:
        raise ValueError("target length cannot hold the fact and filler")
    filler_count = target_tokens - len(fact_ids)
    repeats = (filler_count + len(filler_ids) - 1) // len(filler_ids)
    return (filler_ids * repeats)[:filler_count] + fact_ids


def answer_matches(prediction, expected):
    """Match the first answer line, ignoring case and punctuation."""
    def clean(text):
        return " ".join(re.findall(r"\w+", text.casefold()))
    answer = clean(expected)
    lines = prediction.splitlines()
    first_line = lines[0] if lines else ""
    return bool(answer) and f" {answer} " in f" {clean(first_line)} "
