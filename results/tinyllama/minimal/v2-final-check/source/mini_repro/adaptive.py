"""Repeated local measurements for a causal, offline bandwidth adaptation study."""

import argparse
import json
import math
from pathlib import Path
import random
import statistics

import torch

from mini_repro.data import answer_matches
from mini_repro.experiment_common import build_positioned_case, choose_strategy
from mini_repro.quality_size import load_local_model
from mini_repro.run import (
    MODEL_ID, ROOT, answer_from_cache, blob_to_legacy, legacy_to_blob,
    make_codec, prefill, save_results,
)
from mini_repro.ttft import first_token, timed_gpu


DEFAULT_RESULTS = ROOT / "results" / "tinyllama" / "adaptive.json"
STRATEGIES = ("adaptive", "adaptive_static", "fixed1", "fixed2", "fixed3", "text")


def representation_cost(chunk, level, bandwidth_gbps):
    row = chunk["text"] if level == 0 else chunk["levels"][level]
    transfer = row["bytes"] * 8 / (bandwidth_gbps * 1e9)
    compute = row["prefill_seconds"] if level == 0 else row["decode_seconds"]
    return transfer, compute


def next_decision(chunks, index, bandwidth, slo, elapsed, strategy, first_token_reserve=0.0):
    """Shared causal decision for estimates and actual socket execution."""
    if strategy not in STRATEGIES or not 0 <= index < len(chunks) or bandwidth <= 0:
        raise ValueError('invalid strategy, chunk index or bandwidth')
    chunk = chunks[index]
    remaining = slo - first_token_reserve - elapsed
    forecasts = {level: sum(sum(representation_cost(item, level, bandwidth))
                            for item in chunks[index:]) for level in (3, 2, 1, 0)}
    if strategy == 'adaptive':
        level = next((level for level in (3, 2, 1, 0) if forecasts[level] <= remaining + 1e-12), None)
        if level is None:
            level = min((3, 2, 1, 0), key=lambda candidate: sum(representation_cost(chunk, candidate, bandwidth)))
    elif strategy == 'adaptive_static':
        level = choose_strategy(chunk['levels'], bandwidth, slo / len(chunks))
    else:
        level = 0 if strategy == 'text' else int(strategy[-1])
    return {'chunk_index': index, 'bandwidth_gbps': bandwidth, 'remaining_budget_seconds': remaining,
            'level': level, 'forecast_seconds': forecasts[level],
            'forecast_feasible': forecasts[level] <= remaining + 1e-12}


def plan_chunks(chunks, bandwidths, slo_seconds, strategy="adaptive", first_token_reserve=0.0):
    """Plan using known chunk metadata and only the current observed bandwidth.

    The adaptive heuristic forecasts all remaining chunks at the current
    bandwidth, tries a common CacheGen level 3, 2, then 1, and then text.
    Past estimated transfer and local work consume the remaining deadline;
    the final question/first-token calibration is reserved separately. When
    no forecast fits, use the fastest current representation and expose the
    infeasibility. This is a local heuristic, not the paper's exact scheduler.
    """
    if (strategy not in STRATEGIES or len(bandwidths) < len(chunks)
            or not math.isfinite(slo_seconds) or slo_seconds <= 0 or not chunks
            or not math.isfinite(first_token_reserve) or first_token_reserve < 0):
        raise ValueError("invalid strategy, trace length or SLO")
    levels = []
    decisions = []
    transfer_seconds = 0.0
    compute_seconds = 0.0
    for index, (chunk, bw) in enumerate(zip(chunks, bandwidths)):
        if not math.isfinite(bw) or bw <= 0:
            raise ValueError("bandwidth must be positive")
        decision = next_decision(chunks, index, bw, slo_seconds, transfer_seconds + compute_seconds,
                                 strategy, first_token_reserve)
        level = decision['level']
        transfer, compute = representation_cost(chunk, level, bw)
        transfer_seconds += transfer
        compute_seconds += compute
        levels.append(level)
        decisions.append(decision)
    estimated_ttft = transfer_seconds + compute_seconds + first_token_reserve
    return {"levels": levels, "transfer_seconds": transfer_seconds,
            "premeasured_compute_seconds": compute_seconds,
            "estimated_seconds": transfer_seconds + compute_seconds,
            "first_token_reserve_seconds": first_token_reserve,
            "estimated_ttft_seconds": estimated_ttft,
            "feasible": estimated_ttft <= slo_seconds + 1e-12,
            "decisions": decisions}


def load_bandwidth_traces(path, count, nchunks, scale):
    if count < 1 or scale <= 0:
        raise ValueError("trace count and scale must be positive")
    traces = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                values = json.loads(line)["bw"]
                if len(values) < nchunks:
                    raise ValueError("bandwidth trace shorter than context chunks")
                traces.append([float(value) * scale for value in values[:nchunks]])
            if len(traces) == count:
                break
    if len(traces) < count:
        raise ValueError("not enough bandwidth traces")
    return traces


@torch.inference_mode()
def measure_text_prefill(model, ids, previous_blob):
    inputs = torch.tensor([ids], dtype=torch.long, device="cuda")
    cache = blob_to_legacy(previous_blob) if previous_blob is not None else None
    _, seconds = timed_gpu(lambda: model(input_ids=inputs, past_key_values=cache, use_cache=True))
    return seconds


def prepare_chunks(model, tokenizer, context_ids, raw_blob, chunk_size, repeats=3, warmup=1):
    chunks = []
    for start in range(0, len(context_ids), chunk_size):
        end = min(start + chunk_size, len(context_ids))
        ids = context_ids[start:end]
        chunk_blob = raw_blob[:, :, :, start:end, :]
        text_bytes = len(tokenizer.decode(ids, skip_special_tokens=False,
                                          clean_up_tokenization_spaces=False).encode("utf-8"))
        previous = raw_blob[:, :, :, :start, :] if start else None
        for _ in range(warmup):
            measure_text_prefill(model, ids, previous)
        text_timings = [measure_text_prefill(model, ids, previous) for _ in range(repeats)]
        text_cost = statistics.median(text_timings)
        row = {"start": start, "end": end, "ids": ids, "text": {
            "bytes": text_bytes, "prefill_seconds": text_cost,
            "prefill_calibration_seconds": text_timings},
            "levels": {}, "payloads": {}, "decoders": {}}
        for level in (1, 2, 3):
            encoder, decoder = make_codec(len(ids), level=level)
            payload, encode_s = timed_gpu(lambda: encoder.to_bytes(chunk_blob))
            for _ in range(warmup):
                warmed, _ = timed_gpu(lambda: decoder.from_bytes(payload))
                del warmed
            decode_timings = []
            for _ in range(repeats):
                restored, decode_s = timed_gpu(lambda: decoder.from_bytes(payload))
                decode_timings.append(decode_s)
            decode_s = statistics.median(decode_timings)
            if restored.shape != chunk_blob.shape or not torch.isfinite(restored).all():
                raise RuntimeError(f"invalid chunk codec level {level}")
            row["levels"][level] = {"bytes": len(payload), "decode_seconds": decode_s,
                                    "decode_calibration_seconds": decode_timings,
                                    "offline_encode_seconds": encode_s}
            row["payloads"][level] = payload
            row["decoders"][level] = decoder
            del restored, encoder
        chunks.append(row)
    return chunks


@torch.inference_mode()
def materialize(model, chunks, levels):
    """Build selected KV; text chunks are recomputed using preceding KV."""
    merged = None
    for chunk, level in zip(chunks, levels):
        if level == 0:
            inputs = torch.tensor([chunk["ids"]], dtype=torch.long, device="cuda")
            cache = blob_to_legacy(merged) if merged is not None else None
            output = model(input_ids=inputs, past_key_values=cache, use_cache=True)
            merged = legacy_to_blob(output.past_key_values)
        else:
            decoded = chunk["decoders"][level].from_bytes(chunk["payloads"][level])
            merged = decoded if merged is None else torch.cat((merged, decoded), dim=3)
    return merged


def summarize(rows):
    """Separate timing repetitions, fact/trace QA conditions, and unique facts."""
    result = {}
    for strategy in STRATEGIES:
        selected = [item for case in rows for item in case["measurements"] if item["strategy"] == strategy]
        qa_rows = {}
        for case_index, case in enumerate(rows):
            fact_key = case.get("case_id", case_index)
            for measurement_index, item in enumerate(case["measurements"]):
                if item["strategy"] == strategy:
                    trace_key = item.get("trace_id", measurement_index)
                    qa_rows.setdefault((fact_key, trace_key), item)
        timings = sorted(item["estimated_ttft_seconds"] for item in selected)
        p95_index = (len(timings) - 1) * 0.95 if timings else 0
        p95 = (timings[math.floor(p95_index)] * (1 - (p95_index % 1))
               + timings[math.ceil(p95_index)] * (p95_index % 1)) if timings else None
        result[strategy] = {"runs": len(selected), "qa_cases": len(qa_rows),
                            "qa_conditions": len(qa_rows),
                            "unique_fact_cases": len({fact_key for fact_key, _ in qa_rows}),
                            "qa_count_definition": "qa_cases and qa_conditions count unique fact/trace conditions; unique_fact_cases counts fact contexts; runs counts timing repetitions",
                            "slo_violations": sum(item["slo_violation"] for item in selected),
                            "correct": sum(item["correct"] for item in qa_rows.values()),
                            "qa_correct": sum(item["correct"] for item in qa_rows.values()),
                            "median_estimated_ttft_seconds": statistics.median(timings) if timings else None,
                            "p95_estimated_ttft_seconds": p95,
                            "mean_estimated_ttft_seconds": sum(item["estimated_ttft_seconds"] for item in selected) / len(selected) if selected else None}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=5)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--traces", type=int, default=5)
    parser.add_argument("--bandwidth-scale", type=float, default=1.0)
    parser.add_argument("--slo", type=float, default=0.5)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    if (min(args.cases, args.length, args.chunk_size, args.traces, args.max_new_tokens, args.repeats) < 1
            or args.warmup < 0 or args.cases > 10
            or not math.isfinite(args.slo) or args.slo <= 0):
        parser.error("counts/SLO must be positive, warmup nonnegative, and cases at most 10 unique facts")
    model, tokenizer = load_local_model(seed=args.seed)
    rng = random.Random(args.seed)
    torch.cuda.reset_peak_memory_stats()
    max_question = len(tokenizer.encode("\nQuestion: Who donated the silver compass?\nAnswer:", add_special_tokens=False))
    if args.length + max_question + args.max_new_tokens > model.config.max_position_embeddings:
        parser.error("context plus question exceeds model window")
    chunk_count = (args.length + args.chunk_size - 1) // args.chunk_size
    traces = load_bandwidth_traces(ROOT / "test_data" / "bw.jsonl", args.traces,
                                   chunk_count, args.bandwidth_scale)
    data = {"schema_version": 2, "status": "running", "model": MODEL_ID,
            "measurement": "trace transfer estimate + repeatedly measured local reconstruction and first-token time; no live network",
            "strategies": list(STRATEGIES),
            "policy": "adaptive forecasts remaining known chunk sizes/costs at current bandwidth only, subtracts spent estimated time, and reserves calibrated first-token latency; adaptive_static uses SLO/chunk_count",
            "limitations": [
                "Scheduler is a causal local heuristic, not an exact implementation of the paper scheduler.",
                "Transfer is an estimate without streaming/decode overlap or a network socket.",
                "Text uses original token IDs for recomputation; decoded UTF-8 length is a transfer-size proxy without receive-side retokenization.",
                "QA evaluated once per case/trace/strategy and reused across timing repeats; repeats are not independent QA samples."],
            "parameters": {"cases": args.cases, "length": args.length, "chunk_size": args.chunk_size,
                           "traces": args.traces, "bandwidth_scale": args.bandwidth_scale,
                           "slo_seconds": args.slo, "max_new_tokens": args.max_new_tokens,
                           "repeats": args.repeats, "warmup": args.warmup, "seed": args.seed}, "cases": []}
    save_results(args.results, data)
    for index in range(args.cases):
        position = ("start", "middle", "end")[index % 3]
        case = build_positioned_case(tokenizer, index, args.length, position)
        raw_blob = prefill(model, case["input_ids"])
        chunks = prepare_chunks(model, tokenizer, case["input_ids"], raw_blob, args.chunk_size,
                                repeats=args.repeats, warmup=args.warmup)
        question_ids = tokenizer.encode(f"\nQuestion: {case['question']}\nAnswer:", add_special_tokens=False)
        for _ in range(max(1, args.warmup)):
            first_token(model, case["input_ids"], question_ids, raw_blob)
        first_token_timings = [timed_gpu(lambda: first_token(
            model, case["input_ids"], question_ids, raw_blob))[1] for _ in range(max(3, args.repeats))]
        first_token_reserve = statistics.median(first_token_timings)
        raw_prediction = answer_from_cache(model, tokenizer, raw_blob, case["question"], args.max_new_tokens)
        row = {"case_id": case["case_id"], "position": position, "answer": case["answer"],
               "question": case["question"], "fact_start": case["fact_start"],
               "raw_prediction": raw_prediction, "raw_correct": answer_matches(raw_prediction, case["answer"]),
               "calibration": {"first_token_reserve_seconds": first_token_reserve,
                               "raw_first_token_seconds": first_token_timings,
                               "first_token_warmup": max(1, args.warmup),
                               "chunks": [{key: chunk[key] for key in ("start", "end", "text", "levels")}
                                          for chunk in chunks]},
               "measurements": []}
        for trace_id, bandwidths in enumerate(traces):
            plans = {strategy: plan_chunks(chunks, bandwidths, args.slo, strategy,
                                          first_token_reserve=first_token_reserve)
                     for strategy in STRATEGIES}
            predictions = {}
            for strategy in STRATEGIES:
                plan = plans[strategy]
                for _ in range(args.warmup):
                    merged, _ = timed_gpu(lambda: materialize(model, chunks, plan["levels"]))
                    first_token(model, case["input_ids"], question_ids, merged)
                    del merged
                merged = materialize(model, chunks, plan["levels"])
                if merged.shape != raw_blob.shape:
                    raise RuntimeError("adaptive KV shape differs from original")
                if not torch.isfinite(merged).all():
                    raise RuntimeError("adaptive KV contains non-finite values")
                predictions[strategy] = answer_from_cache(model, tokenizer, merged,
                                                         case["question"], args.max_new_tokens)
                del merged
            for repeat in range(args.repeats):
                strategy_order = list(STRATEGIES)
                rng.shuffle(strategy_order)
                for order_index, strategy in enumerate(strategy_order):
                    plan = plans[strategy]
                    merged, reconstruct_s = timed_gpu(lambda: materialize(model, chunks, plan["levels"]))
                    _, first_s = timed_gpu(lambda: first_token(model, case["input_ids"], question_ids, merged))
                    estimated_ttft = plan["transfer_seconds"] + reconstruct_s + first_s
                    prediction = predictions[strategy]
                    row["measurements"].append({
                        "trace_id": trace_id, "strategy": strategy, "repeat": repeat,
                        "order_in_repeat": order_index, "bandwidth_gbps": bandwidths,
                        "levels": plan["levels"], "decisions": plan["decisions"],
                        "first_token_reserve_seconds": first_token_reserve,
                        "plan_feasible": plan["feasible"],
                        "decision_forecasts_feasible": all(item["forecast_feasible"] for item in plan["decisions"]),
                        "premeasured_estimated_ttft_seconds": plan["estimated_ttft_seconds"],
                        "estimated_transfer_seconds": round(plan["transfer_seconds"], 6),
                        "local_reconstruct_seconds": round(reconstruct_s, 6),
                        "local_first_token_seconds": round(first_s, 6),
                        "estimated_ttft_seconds": round(estimated_ttft, 6),
                        "slo_violation": estimated_ttft > args.slo,
                        "prediction": prediction, "correct": answer_matches(prediction, case["answer"]),
                    })
                    del merged
        data["cases"].append(row)
        save_results(args.results, data)
        print(f"adaptive {row['case_id']}", flush=True)
        del raw_blob, chunks
        torch.cuda.empty_cache()
    data["summary"] = summarize(data["cases"])
    data["status"] = "complete"
    data["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
    save_results(args.results, data)
    print(f"Results: {args.results}", flush=True)


if __name__ == "__main__":
    main()
