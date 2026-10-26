"""Single-host, rate-limited transfer and first-token timing experiment."""

import argparse
import math
import random
import socket
import threading
import time
from collections import defaultdict
from pathlib import Path
from statistics import median

import torch

from mini_repro.experiment_common import load_longchat_segments, pack_uniform, unpack_uniform
from mini_repro.quality_size import load_local_model
from mini_repro.run import MODEL_ID, ROOT, blob_to_legacy, make_codec, prefill, save_results


DEFAULT_RESULTS = ROOT / "results" / "tinyllama" / "ttft.json"
QUESTION = "\nQuestion: What is the main topic?\nAnswer:"


# Historical import path; transport is shared by all v2 tasks.
from mini_repro.transport import rate_limited_transfer


def timed_gpu(function):
    torch.cuda.synchronize()
    start = time.perf_counter()
    value = function()
    torch.cuda.synchronize()
    return value, time.perf_counter() - start


def measurement_schedule(methods, bandwidths, *, warmup, repeats, seed):
    """Balance every method per repetition while randomizing execution order."""
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be nonnegative and repeats must be positive")
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("methods must be nonempty and unique")
    if not bandwidths or any(not math.isfinite(value) or value <= 0 for value in bandwidths):
        raise ValueError("bandwidths must be positive and finite")
    if len(set(bandwidths)) != len(bandwidths):
        raise ValueError("bandwidths must be unique")
    rng = random.Random(seed)
    schedule = []
    for bandwidth in bandwidths:
        for phase, count in (("warmup", warmup), ("measure", repeats)):
            for repetition in range(1, count + 1):
                order = list(methods)
                rng.shuffle(order)
                for position, method in enumerate(order):
                    schedule.append({"phase": phase, "repetition": repetition,
                                     "method_order": position, "method": method,
                                     "bandwidth_gbps": bandwidth})
    return schedule


def summarize_ttft(cases):
    """Pooled timing median/P95, with context and repetition counts separated."""
    groups = defaultdict(list)
    context_counts = defaultdict(lambda: defaultdict(int))
    for index, case in enumerate(cases):
        context = case.get("case_id", f"context-{index}")
        for row in case["measurements"]:
            key = f"{row['method']}@{row['bandwidth_gbps']:g}Gbps@{case['context_tokens']}tokens"
            groups[key].append(row["ttft_wall_seconds"])
            context_counts[key][context] += 1
    result = {}
    for key, values in groups.items():
        ordered = sorted(values)
        rank = (len(ordered) - 1) * 0.95
        lower = int(rank)
        upper = min(lower + 1, len(ordered) - 1)
        p95 = ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)
        result[key] = {"samples": len(ordered), "timing_repetitions": len(ordered),
                       "contexts": len(context_counts[key]),
                       "repetitions_per_context": dict(context_counts[key]),
                       "aggregation": "pooled timings across contexts and repetitions",
                       "median_seconds": round(median(ordered), 6),
                       "p95_seconds": round(p95, 6)}
    return result


@torch.inference_mode()
def first_token(model, context_ids, question_ids, cache=None):
    ids = question_ids if cache is not None else context_ids + question_ids
    inputs = torch.tensor([ids], dtype=torch.long, device="cuda")
    if cache is None:
        logits = model(input_ids=inputs, use_cache=True).logits
    else:
        logits = model(input_ids=inputs, past_key_values=blob_to_legacy(cache), use_cache=True).logits
    return int(logits[0, -1].argmax().item())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--lengths", type=int, nargs="+", default=[512, 1024, 1536])
    parser.add_argument("--bandwidth-gbps", type=float, nargs="+", default=[0.1, 1.0, 10.0])
    parser.add_argument("--cachegen-levels", type=int, nargs="+", choices=[1, 2, 3], default=[2, 3])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    if args.samples < 1 or min(args.lengths) < 1:
        parser.error("samples and lengths must be positive")
    methods = ["text", "uniform8"] + [f"cachegen{level}" for level in args.cachegen_levels]
    try:
        measurement_schedule(methods, args.bandwidth_gbps, warmup=args.warmup,
                             repeats=args.repeats, seed=args.seed)
    except ValueError as error:
        parser.error(str(error))
    model, tokenizer = load_local_model(seed=args.seed)
    question_ids = tokenizer.encode(QUESTION, add_special_tokens=False)
    if max(args.lengths) + len(question_ids) > model.config.max_position_embeddings:
        parser.error("context plus question exceeds model window")
    cases = load_longchat_segments(ROOT / "test_data" / "longchat.jsonl", tokenizer,
                                   lengths=args.lengths, sample_count=args.samples, continuation_tokens=2)
    data = {"model": MODEL_ID,
            "measurement": "single-host paced socket; receive complete payload, then reconstruct and generate first token; offline encoding excluded",
            "scope": "existing repository codec on TinyLlama; no streamed decode overlap or paper-scale model",
            "summary_definition": "median/P95 pool repeated timings; contexts counts independent source prefixes, not repetitions",
            "methods": methods,
            "parameters": {"samples": args.samples, "lengths": args.lengths,
                           "bandwidth_gbps": args.bandwidth_gbps,
                           "cachegen_levels": args.cachegen_levels, "warmup": args.warmup,
                           "repeats": args.repeats, "seed": args.seed}, "cases": []}
    save_results(args.results, data)
    for case_index, case in enumerate(cases):
        context_ids = case["context_ids"]
        text = tokenizer.decode(context_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if tokenizer.encode(text, add_special_tokens=False) != context_ids:
            raise RuntimeError(f"text token roundtrip differs for {case['case_id']}; cannot compare identical contexts")
        blob = prefill(model, context_ids)
        uniform_payload, uniform_encode_s = timed_gpu(lambda: pack_uniform(blob, bits=8))
        payloads = {
            "text": (text.encode("utf-8"), 0.0, None),
            "uniform8": (uniform_payload, uniform_encode_s, unpack_uniform),
        }
        for level in args.cachegen_levels:
            encoder, decoder = make_codec(len(context_ids), level=level)
            payload, encode_s = timed_gpu(lambda: encoder.to_bytes(blob))
            payloads[f"cachegen{level}"] = (payload, encode_s, decoder.from_bytes)
        row = {"case_id": case["case_id"], "source_id": case["source_id"],
               "context_tokens": len(context_ids), "schedule_seed": args.seed + case_index,
               "warmup_measurements": args.warmup * len(methods) * len(args.bandwidth_gbps),
               "measurements": []}
        schedule = measurement_schedule(methods, args.bandwidth_gbps, warmup=args.warmup,
                                        repeats=args.repeats, seed=row["schedule_seed"])
        for step in schedule:
            method, bandwidth = step["method"], step["bandwidth_gbps"]
            payload, offline_encode_s, restore = payloads[method]
            start = time.perf_counter()
            received, transfer_s = rate_limited_transfer(payload, bandwidth)
            if method == "text":
                reconstruct_start = time.perf_counter()
                text_ids = tokenizer.encode(received.decode("utf-8"), add_special_tokens=False)
                reconstruction_s = time.perf_counter() - reconstruct_start
                if text_ids != context_ids:
                    raise RuntimeError("text changed after transport")
                token, first_s = timed_gpu(lambda: first_token(model, text_ids, question_ids))
            else:
                restored, reconstruction_s = timed_gpu(lambda: restore(received))
                token, first_s = timed_gpu(lambda: first_token(model, context_ids, question_ids, restored))
                del restored
            wall_s = time.perf_counter() - start
            if step["phase"] == "measure":
                realized_bandwidth = len(payload) * 8 / transfer_s / 1e9
                row["measurements"].append({
                    "method": method, "bandwidth_gbps": bandwidth,
                    "repetition": step["repetition"], "method_order": step["method_order"],
                    "bytes": len(payload), "offline_encode_seconds": round(offline_encode_s, 6),
                    "transfer_seconds": round(transfer_s, 6),
                    "realized_bandwidth_gbps": round(realized_bandwidth, 6),
                    "realized_to_target_bandwidth_ratio": round(realized_bandwidth / bandwidth, 6),
                    "reconstruct_seconds": round(reconstruction_s, 6),
                    "reconstruct_kind": "tokenize" if method == "text" else "decode",
                    "first_token_seconds": round(first_s, 6), "first_token_id": token,
                    "ttft_wall_seconds": round(wall_s, 6),
                })
        text_tokens = {item["first_token_id"] for item in row["measurements"] if item["method"] == "text"}
        if len(text_tokens) != 1:
            raise RuntimeError(f"text baseline generated inconsistent first tokens for {case['case_id']}")
        baseline_token = text_tokens.pop()
        row["text_first_token_id"] = baseline_token
        for item in row["measurements"]:
            item["matches_text_first_token"] = item["first_token_id"] == baseline_token
        data["cases"].append(row)
        save_results(args.results, data)
        print(f"ttft {row['case_id']}", flush=True)
        del blob, uniform_payload, payloads, encoder, decoder, payload, restore, received
        torch.cuda.empty_cache()
    data["summary"] = summarize_ttft(data["cases"])
    data["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
    save_results(args.results, data)
    print(f"Results: {args.results}", flush=True)


if __name__ == "__main__":
    main()
