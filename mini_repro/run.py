"""Small, offline TinyLlama check of the author's CacheGen GPU codec."""

import argparse
import io
import json
import os
import platform
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mini_repro.data import QA_CASES, answer_matches, build_qa_context, load_longchat_prefixes


MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results" / "tinyllama" / "metrics.json"


def serialized_size(value):
    """Count a transportable torch container, including its metadata."""
    buffer = io.BytesIO()
    if isinstance(value, torch.Tensor):
        value = value.cpu()
    else:
        value = tuple(item.cpu() if isinstance(item, torch.Tensor) else item for item in value)
    torch.save(value, buffer)
    return buffer.tell()


def uniform8_roundtrip(blob):
    """Author's 256-bin, per-token/channel-group quantization formula."""
    layers, kinds, heads, tokens, width = blob.shape
    groups = blob.permute(0, 1, 3, 2, 4).reshape(layers, kinds, tokens, heads * width)
    maxima = groups.abs().amax(dim=-1, keepdim=True)
    safe_maxima = maxima.clamp_min(torch.finfo(maxima.dtype).tiny)
    quantized = torch.round(groups * (127.0 / safe_maxima)).clamp(-127, 127).to(torch.int8)
    restored = (quantized.to(blob.dtype) / 127.0 * maxima).reshape(layers, kinds, tokens, heads, width)
    restored = restored.permute(0, 1, 3, 2, 4).to(blob.dtype)
    return restored, serialized_size((quantized, maxima))


def blob_to_legacy(blob):
    return tuple((blob[layer, 0].unsqueeze(0), blob[layer, 1].unsqueeze(0)) for layer in range(blob.shape[0]))


def legacy_to_blob(cache):
    if hasattr(cache, "to_legacy_cache"):
        cache = cache.to_legacy_cache()
    return torch.stack([torch.stack((key[0], value[0]), dim=0) for key, value in cache], dim=0)


def make_codec(tokens, level=2):
    from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
    from lmcache.storage_backend.serde.cachegen_encoder import CacheGenSerializer
    from lmcache.storage_backend.serde.cachegen_decoder import CacheGenDeserializer

    if level not in (1, 2, 3):
        raise ValueError("CacheGen level must be 1, 2 or 3")
    config = LMCacheEngineConfig.from_defaults(chunk_size=tokens)
    metadata = LMCacheEngineMetadata(MODEL_ID, 1, 0, "huggingface")
    previous = os.environ.get("QUANT_LEVEL")
    try:
        os.environ["QUANT_LEVEL"] = str(level)
        return CacheGenSerializer(config, metadata), CacheGenDeserializer(config, metadata)
    finally:
        if previous is None:
            os.environ.pop("QUANT_LEVEL", None)
        else:
            os.environ["QUANT_LEVEL"] = previous


def timed_cuda(function):
    torch.cuda.synchronize()
    start = time.perf_counter()
    value = function()
    torch.cuda.synchronize()
    return value, round(time.perf_counter() - start, 4)


@torch.inference_mode()
def prefill(model, token_ids):
    inputs = torch.tensor([token_ids], dtype=torch.long, device="cuda")
    return legacy_to_blob(model(input_ids=inputs, use_cache=True).past_key_values)


@torch.inference_mode()
def answer_from_cache(model, tokenizer, blob, question, max_new_tokens=24):
    question_ids = tokenizer.encode(f"\nQuestion: {question}\nAnswer:", add_special_tokens=False)
    if blob.shape[-2] + len(question_ids) + max_new_tokens > model.config.max_position_embeddings:
        raise ValueError("prompt exceeds model context window")
    inputs = torch.tensor([question_ids], dtype=torch.long, device="cuda")
    result = model(input_ids=inputs, past_key_values=blob_to_legacy(blob), use_cache=True)
    generated = []
    for step in range(max_new_tokens):
        next_id = result.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        token = int(next_id.item())
        if token == tokenizer.eos_token_id:
            break
        generated.append(token)
        if step + 1 < max_new_tokens:
            result = model(input_ids=next_id, past_key_values=result.past_key_values, use_cache=True)
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def save_results(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--compression-limit", type=int, default=6)
    parser.add_argument("--qa-limit", type=int, default=5)
    args = parser.parse_args()
    if not 0 <= args.compression_limit <= 6 or not 0 <= args.qa_limit <= len(QA_CASES):
        parser.error("limits must be within the six compression and five QA cases")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the author's CacheGen codec")

    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, local_files_only=True, torch_dtype=torch.float16, low_cpu_mem_usage=False
    ).to("cuda").eval()
    cases = load_longchat_prefixes(ROOT / "test_data" / "longchat.jsonl", tokenizer)
    results = {
        "model": MODEL_ID,
        "method": "author CacheGen GPU codec, exploratory TinyLlama layer schedule",
        "uniform_baseline": "256-bin per-token/channel-group quantization from src/utils.py, torch.save container",
        "dataset": "first three bundled LongChat prompts at 512 and 1024 tokens; five local synthetic fact QA prompts",
        "qa_scoring": "case-insensitive exact answer phrase in the first generated line",
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "hf_home": os.environ.get("HF_HOME"),
        },
        "compression": [],
        "qa": [],
    }
    save_results(args.results, results)

    for case in cases[:args.compression_limit]:
        blob = prefill(model, case["input_ids"])
        raw_bytes = serialized_size(blob)
        uniform, uniform_bytes = uniform8_roundtrip(blob)
        serializer, deserializer = make_codec(case["length"])
        encoded, encode_seconds = timed_cuda(lambda: serializer.to_bytes(blob))
        restored, decode_seconds = timed_cuda(lambda: deserializer.from_bytes(encoded))
        if restored.shape != blob.shape or not torch.isfinite(restored).all():
            raise RuntimeError(f"invalid codec output for {case['case_id']}")
        row = {
            "case_id": case["case_id"],
            "source_id": case["source_id"],
            "tokens": case["length"],
            "kv_shape": list(blob.shape),
            "raw_bytes": raw_bytes,
            "uniform8_bytes": uniform_bytes,
            "cachegen_bytes": len(encoded),
            "raw_over_cachegen": round(raw_bytes / len(encoded), 3),
            "uniform8_over_cachegen": round(uniform_bytes / len(encoded), 3),
            "uniform8_mae": round((uniform - blob).abs().float().mean().item(), 7),
            "cachegen_mae": round((restored - blob).abs().float().mean().item(), 7),
            "encode_seconds": encode_seconds,
            "decode_seconds": decode_seconds,
        }
        results["compression"].append(row)
        save_results(args.results, results)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        del blob, uniform, serializer, deserializer, encoded, restored
        torch.cuda.empty_cache()

    for case_id, fact, question, expected in QA_CASES[:args.qa_limit]:
        context = build_qa_context(tokenizer, fact, target_tokens=512)
        blob = prefill(model, context)
        serializer, deserializer = make_codec(len(context))
        encoded = serializer.to_bytes(blob)
        restored = deserializer.from_bytes(encoded)
        uniform, _ = uniform8_roundtrip(blob)
        predictions = {
            "raw": answer_from_cache(model, tokenizer, blob, question),
            "uniform8": answer_from_cache(model, tokenizer, uniform, question),
            "cachegen": answer_from_cache(model, tokenizer, restored, question),
        }
        row = {
            "case_id": case_id,
            "context_tokens": len(context),
            "question": question,
            "expected": expected,
            "predictions": predictions,
            "correct": {name: answer_matches(text, expected) for name, text in predictions.items()},
            "cachegen_bytes": len(encoded),
        }
        results["qa"].append(row)
        save_results(args.results, results)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        del blob, uniform, restored, serializer, deserializer, encoded
        torch.cuda.empty_cache()

    results["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
    save_results(args.results, results)
    print(f"Results: {args.results}", flush=True)


if __name__ == "__main__":
    main()
