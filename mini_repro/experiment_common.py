"""Shared offline experiment data, quantization, and selection helpers."""

import io
import json
from pathlib import Path

import torch

from mini_repro.data import FILLER


DONORS = ("Clara", "Maya", "Nora", "Ethan", "Priya", "Louis", "Elena", "Jonah", "Iris", "Felix")
OBJECTS = ("blue vase", "winter flowers", "river atlas", "silver compass", "antique clock",
           "ceramic bowl", "star map", "red notebook", "garden key", "glass lantern")


def load_longchat_segments(path, tokenizer, lengths=(512, 1024, 1536), sample_count=20, continuation_tokens=32):
    """Select fixed context and genuinely unseen continuation tokens."""
    if sample_count < 1 or continuation_tokens < 2 or any(length < 1 for length in lengths):
        raise ValueError("sample_count, lengths and continuation_tokens must be positive")
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                token_ids = tokenizer.encode(row["prompt"], add_special_tokens=False, verbose=False)
                rows.append((row["test_id"], token_ids))
            if len(rows) >= sample_count:
                break
    if len(rows) < sample_count:
        raise ValueError("not enough LongChat rows")
    cases = []
    for length in lengths:
        for source_id, ids in rows:
            if len(ids) < length + continuation_tokens:
                raise ValueError(f"LongChat row {source_id} is too short")
            cases.append({
                "case_id": f"longchat-{source_id}-{length}",
                "source_id": source_id,
                "context_ids": ids[:length],
                "continuation_ids": ids[length:length + continuation_tokens],
            })
    return cases


def build_positioned_case(tokenizer, index, length, position):
    """One deterministic answer-bearing fact at a controlled token position."""
    if position not in {"start", "middle", "end"}:
        raise ValueError("position must be start, middle or end")
    answer = DONORS[index % len(DONORS)]
    item = OBJECTS[index % len(OBJECTS)]
    fact = f" In the museum log, the {item} was donated by {answer}. "
    fact_ids = tokenizer.encode(fact, add_special_tokens=False)
    filler_ids = tokenizer.encode(FILLER, add_special_tokens=False)
    if length < len(fact_ids) or not filler_ids:
        raise ValueError("context length too short for the fact")
    room = length - len(fact_ids)
    start = {"start": 0, "middle": room // 2, "end": room}[position]
    filler = (filler_ids * ((length // len(filler_ids)) + 2))[:length]
    ids = filler[:start] + fact_ids + filler[:room - start]
    return {
        "case_id": f"fact-{index}-{length}-{position}",
        "input_ids": ids,
        "question": f"Who donated the {item}?",
        "answer": answer,
        "position": position,
        "fact_start": start,
    }


def pack_uniform(blob, bits=8):
    """Serialize per-token, per-layer uniform quantization including scales."""
    if bits not in (4, 8):
        raise ValueError("uniform bits must be 4 or 8")
    layers, kinds, heads, tokens, width = blob.shape
    groups = blob.permute(0, 1, 3, 2, 4).reshape(layers, kinds, tokens, heads * width)
    maxima = groups.abs().amax(dim=-1, keepdim=True)
    qmax = (1 << (bits - 1)) - 1
    safe = maxima.clamp_min(torch.finfo(maxima.dtype).tiny)
    quantized = torch.round(groups.float() / safe.float() * qmax).clamp(-qmax, qmax).to(torch.int8).flatten()
    if bits == 4:
        symbols = (quantized.to(torch.int16) + qmax).to(torch.uint8)
        if symbols.numel() % 2:
            symbols = torch.cat((symbols, torch.zeros(1, device=symbols.device, dtype=torch.uint8)))
        values = (symbols[0::2] | (symbols[1::2] << 4)).cpu()
    else:
        values = quantized.cpu()
    stream = io.BytesIO()
    torch.save({"bits": bits, "shape": tuple(blob.shape), "count": quantized.numel(),
                "values": values, "maxima": maxima.cpu()}, stream)
    return stream.getvalue()


def unpack_uniform(payload, device="cuda"):
    """Reconstruct FP16/FP32 KV from a transmitted uniform payload."""
    saved = torch.load(io.BytesIO(payload), map_location=device, weights_only=True)
    bits = saved["bits"]
    qmax = (1 << (bits - 1)) - 1
    values = saved["values"]
    if bits == 4:
        low = (values & 15).to(torch.int16)
        high = ((values >> 4) & 15).to(torch.int16)
        symbols = torch.stack((low, high), dim=-1).flatten()[:saved["count"]]
        quantized = symbols - qmax
    elif bits == 8:
        quantized = values
    else:
        raise ValueError(f"unsupported bits: {bits}")
    layers, kinds, heads, tokens, width = saved["shape"]
    maxima = saved["maxima"]
    groups = quantized.reshape(layers, kinds, tokens, heads * width).to(maxima.dtype)
    restored = groups / qmax * maxima
    return restored.reshape(layers, kinds, tokens, heads, width).permute(0, 1, 3, 2, 4)


def choose_strategy(level_stats, bandwidth_gbps, budget_seconds):
    """Choose the highest-quality CacheGen level meeting a chunk deadline."""
    if bandwidth_gbps <= 0 or budget_seconds <= 0:
        raise ValueError("bandwidth and budget must be positive")
    for level in (3, 2, 1):
        if level in level_stats:
            row = level_stats[level]
            delay = row["bytes"] * 8 / (bandwidth_gbps * 1e9) + row["decode_seconds"]
            if delay <= budget_seconds:
                return level
    return 0
