"""Offline paired quality-versus-size evaluation for TinyLlama CacheGen."""

import argparse
import math
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from mini_repro.data import answer_matches
from mini_repro.experiment_common import (
    build_positioned_case, load_longchat_segments, pack_uniform, unpack_uniform,
)
from mini_repro.run import (
    MODEL_ID, MODEL_REVISION, ROOT, answer_from_cache, blob_to_legacy, make_codec, prefill,
    save_results, serialized_size, timed_cuda,
)


DEFAULT_RESULTS = ROOT / "results" / "tinyllama" / "quality_size.json"
METHODS = ("raw", "uniform4", "uniform8", "cachegen1", "cachegen2", "cachegen3")


def shifted_nll(logits, token_ids):
    """Mean NLL for suffix tokens 2..N, conditioned on their previous token."""
    if len(token_ids) < 2 or logits.shape[1] != len(token_ids):
        raise ValueError("need matching logits and at least two continuation tokens")
    targets = torch.tensor(token_ids[1:], device=logits.device, dtype=torch.long)
    return F.cross_entropy(logits[0, :-1].float(), targets).item()


@torch.inference_mode()
def continuation_nll(model, blob, continuation_ids):
    if blob.shape[-2] + len(continuation_ids) > model.config.max_position_embeddings:
        raise ValueError("context plus continuation exceeds model window")
    inputs = torch.tensor([continuation_ids], dtype=torch.long, device="cuda")
    cache = DynamicCache.from_legacy_cache(blob_to_legacy(blob))
    logits = model(input_ids=inputs, past_key_values=cache, use_cache=False).logits
    return shifted_nll(logits, continuation_ids)


def load_local_model(seed=0):
    if not torch.cuda.is_available():
        raise RuntimeError("author CacheGen codec requires CUDA")
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True, torch_dtype=torch.float16, low_cpu_mem_usage=False
    ).to("cuda").eval()
    return model, tokenizer


def method_caches(blob):
    """Yield identical-context baselines with transmitted byte counts."""
    yield "raw", blob, serialized_size(blob), 0.0, 0.0
    for bits in (4, 8):
        payload, encode_s = timed_cuda(lambda: pack_uniform(blob, bits))
        restored, decode_s = timed_cuda(lambda: unpack_uniform(payload))
        yield f"uniform{bits}", restored, len(payload), encode_s, decode_s
        del restored, payload
    for level in (1, 2, 3):
        encoder, decoder = make_codec(blob.shape[-2], level=level)
        payload, encode_s = timed_cuda(lambda: encoder.to_bytes(blob))
        restored, decode_s = timed_cuda(lambda: decoder.from_bytes(payload))
        if restored.shape != blob.shape or not torch.isfinite(restored).all():
            raise RuntimeError(f"invalid CacheGen level {level} output")
        yield f"cachegen{level}", restored, len(payload), encode_s, decode_s
        del restored, payload, encoder, decoder


def summarize(data):
    natural = defaultdict(list)
    qa = defaultdict(list)
    for case in data["continuation"]:
        for row in case["methods"]:
            natural[row["method"]].append(row)
    for case in data["qa"]:
        for row in case["methods"]:
            qa[row["method"]].append(row)
    raw_correct = {case["case_id"] for case in data["qa"]
                   if next(row["correct"] for row in case["methods"] if row["method"] == "raw")}
    summary = {}
    for method in METHODS:
        nrows, qrows = natural[method], qa[method]
        summary[method] = {
            "continuation_cases": len(nrows),
            "mean_nll": sum(row["nll"] for row in nrows) / len(nrows) if nrows else None,
            "mean_perplexity_proxy": math.exp(sum(row["nll"] for row in nrows) / len(nrows)) if nrows else None,
            "mean_bytes": sum(row["bytes"] for row in nrows) / len(nrows) if nrows else None,
            "qa_cases": len(qrows),
            "qa_correct": sum(row["correct"] for row in qrows),
            "qa_correct_when_raw_correct": sum(row["correct"] for case in data["qa"]
                                               if case["case_id"] in raw_correct
                                               for row in case["methods"] if row["method"] == method),
            "raw_correct_cases": len(raw_correct),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=20, help="bundled LongChat rows")
    parser.add_argument("--lengths", type=int, nargs="+", default=[512, 1024, 1536])
    parser.add_argument("--continuation", type=int, default=32)
    parser.add_argument("--qa-per-position", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    if args.samples < 0 or args.qa_per_position < 0 or (args.samples == 0 and args.qa_per_position == 0):
        parser.error("request at least one continuation or QA case")
    if min(args.lengths) < 1 or args.continuation < 2 or args.max_new_tokens < 1:
        parser.error("lengths and token counts must be positive")
    model, tokenizer = load_local_model(seed=args.seed)
    if max(args.lengths) + max(args.continuation, args.max_new_tokens + 64) > model.config.max_position_embeddings:
        parser.error("selected context plus suffix may exceed TinyLlama's window")
    data = {
        "model": MODEL_ID,
        "scope": "offline TinyLlama exploratory schedules; not paper-calibrated",
        "nll_definition": "teacher-forced continuation tokens 2..N; first held-out token omitted",
        "methods": list(METHODS), "parameters": vars(args).copy(), "continuation": [], "qa": [],
    }
    data["parameters"]["results"] = str(args.results)
    save_results(args.results, data)
    if args.samples:
        cases = load_longchat_segments(ROOT / "test_data" / "longchat.jsonl", tokenizer,
                                       lengths=args.lengths, sample_count=args.samples,
                                       continuation_tokens=args.continuation)
        for case in cases:
            blob = prefill(model, case["context_ids"])
            row = {"case_id": case["case_id"], "source_id": case["source_id"],
                   "context_tokens": len(case["context_ids"]),
                   "continuation_tokens_scored": len(case["continuation_ids"]) - 1,
                   "methods": []}
            for method, restored, byte_count, encode_s, decode_s in method_caches(blob):
                row["methods"].append({"method": method, "bytes": byte_count,
                                       "nll": round(continuation_nll(model, restored, case["continuation_ids"]), 6),
                                       "kv_mae": round((restored - blob).abs().float().mean().item(), 6),
                                       "offline_encode_seconds": encode_s, "decode_seconds": decode_s})
            data["continuation"].append(row)
            save_results(args.results, data)
            print(f"continuation {row['case_id']}", flush=True)
            del blob
            torch.cuda.empty_cache()
    for length in args.lengths:
        for position in ("start", "middle", "end"):
            for index in range(args.qa_per_position):
                case = build_positioned_case(tokenizer, index, length, position)
                blob = prefill(model, case["input_ids"])
                row = {"case_id": case["case_id"], "context_tokens": length,
                       "position": position, "fact_start": case["fact_start"],
                       "answer": case["answer"], "methods": []}
                for method, restored, byte_count, encode_s, decode_s in method_caches(blob):
                    prediction = answer_from_cache(model, tokenizer, restored, case["question"], args.max_new_tokens)
                    row["methods"].append({"method": method, "bytes": byte_count,
                                           "prediction": prediction,
                                           "correct": answer_matches(prediction, case["answer"]),
                                           "offline_encode_seconds": encode_s, "decode_seconds": decode_s})
                data["qa"].append(row)
                save_results(args.results, data)
                print(f"qa {row['case_id']}", flush=True)
                del blob
                torch.cuda.empty_cache()
    data["summary"] = summarize(data)
    data["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
    save_results(args.results, data)
    print(f"Results: {args.results}", flush=True)


if __name__ == "__main__":
    main()
