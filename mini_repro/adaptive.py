"""Offline trace-based CacheGen level selection and text fallback check."""

import argparse
import json
from pathlib import Path

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
STRATEGIES = ("adaptive", "fixed1", "fixed2", "fixed3", "text")


def plan_chunks(chunks, bandwidths, slo_seconds, strategy="adaptive"):
    """Choose chunk representations and estimate transfer plus local work."""
    if strategy not in STRATEGIES or len(bandwidths) < len(chunks) or slo_seconds <= 0 or not chunks:
        raise ValueError("invalid strategy, trace length or SLO")
    levels = []
    transfer_seconds = 0.0
    compute_seconds = 0.0
    for chunk, bw in zip(chunks, bandwidths):
        if bw <= 0:
            raise ValueError("bandwidth must be positive")
        if strategy == "adaptive":
            level = choose_strategy(chunk["levels"], bw, slo_seconds / len(chunks))
        elif strategy == "text":
            level = 0
        else:
            level = int(strategy[-1])
        row = chunk["text"] if level == 0 else chunk["levels"][level]
        transfer_seconds += row["bytes"] * 8 / (bw * 1e9)
        compute_seconds += row["prefill_seconds"] if level == 0 else row["decode_seconds"]
        levels.append(level)
    return {"levels": levels, "transfer_seconds": transfer_seconds,
            "premeasured_compute_seconds": compute_seconds,
            "estimated_seconds": transfer_seconds + compute_seconds}


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


def prepare_chunks(model, tokenizer, context_ids, raw_blob, chunk_size):
    chunks = []
    for start in range(0, len(context_ids), chunk_size):
        end = min(start + chunk_size, len(context_ids))
        ids = context_ids[start:end]
        chunk_blob = raw_blob[:, :, :, start:end, :]
        text_bytes = len(tokenizer.decode(ids, skip_special_tokens=False,
                                          clean_up_tokenization_spaces=False).encode("utf-8"))
        previous = raw_blob[:, :, :, :start, :] if start else None
        text_cost = measure_text_prefill(model, ids, previous)
        row = {"start": start, "end": end, "ids": ids, "text": {
            "bytes": text_bytes, "prefill_seconds": text_cost},
            "levels": {}, "payloads": {}, "decoders": {}}
        for level in (1, 2, 3):
            encoder, decoder = make_codec(len(ids), level=level)
            payload, _ = timed_gpu(lambda: encoder.to_bytes(chunk_blob))
            restored, decode_s = timed_gpu(lambda: decoder.from_bytes(payload))
            if restored.shape != chunk_blob.shape or not torch.isfinite(restored).all():
                raise RuntimeError(f"invalid chunk codec level {level}")
            row["levels"][level] = {"bytes": len(payload), "decode_seconds": decode_s}
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
    result = {}
    for strategy in STRATEGIES:
        selected = [item for case in rows for item in case["measurements"] if item["strategy"] == strategy]
        result[strategy] = {"runs": len(selected),
                            "slo_violations": sum(item["slo_violation"] for item in selected),
                            "correct": sum(item["correct"] for item in selected),
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
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    if min(args.cases, args.length, args.chunk_size, args.traces, args.max_new_tokens) < 1 or args.slo <= 0:
        parser.error("counts and SLO must be positive")
    model, tokenizer = load_local_model()
    max_question = len(tokenizer.encode("\nQuestion: Who donated the silver compass?\nAnswer:", add_special_tokens=False))
    if args.length + max_question + args.max_new_tokens > model.config.max_position_embeddings:
        parser.error("context plus question exceeds model window")
    chunk_count = (args.length + args.chunk_size - 1) // args.chunk_size
    traces = load_bandwidth_traces(ROOT / "test_data" / "bw.jsonl", args.traces,
                                   chunk_count, args.bandwidth_scale)
    data = {"model": MODEL_ID, "measurement": "trace transfer estimate + locally measured reconstruction and first-token time; no live network",
            "parameters": {"cases": args.cases, "length": args.length, "chunk_size": args.chunk_size,
                           "traces": args.traces, "bandwidth_scale": args.bandwidth_scale,
                           "slo_seconds": args.slo}, "cases": []}
    save_results(args.results, data)
    for index in range(args.cases):
        position = ("start", "middle", "end")[index % 3]
        case = build_positioned_case(tokenizer, index, args.length, position)
        raw_blob = prefill(model, case["input_ids"])
        chunks = prepare_chunks(model, tokenizer, case["input_ids"], raw_blob, args.chunk_size)
        question_ids = tokenizer.encode(f"\nQuestion: {case['question']}\nAnswer:", add_special_tokens=False)
        raw_prediction = answer_from_cache(model, tokenizer, raw_blob, case["question"], args.max_new_tokens)
        row = {"case_id": case["case_id"], "position": position, "answer": case["answer"],
               "raw_prediction": raw_prediction, "raw_correct": answer_matches(raw_prediction, case["answer"]),
               "measurements": []}
        for trace_id, bandwidths in enumerate(traces):
            for strategy in STRATEGIES:
                plan = plan_chunks(chunks, bandwidths, args.slo, strategy)
                merged, reconstruct_s = timed_gpu(lambda: materialize(model, chunks, plan["levels"]))
                if merged.shape != raw_blob.shape:
                    raise RuntimeError("adaptive KV shape differs from original")
                _, first_s = timed_gpu(lambda: first_token(model, case["input_ids"], question_ids, merged))
                prediction = answer_from_cache(model, tokenizer, merged, case["question"], args.max_new_tokens)
                estimated_ttft = plan["transfer_seconds"] + reconstruct_s + first_s
                row["measurements"].append({
                    "trace_id": trace_id, "strategy": strategy, "bandwidth_gbps": bandwidths,
                    "levels": plan["levels"], "estimated_transfer_seconds": round(plan["transfer_seconds"], 6),
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
    data["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
    save_results(args.results, data)
    print(f"Results: {args.results}", flush=True)


if __name__ == "__main__":
    main()
