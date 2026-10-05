"""Single-host, rate-limited transfer and first-token timing experiment."""

import argparse
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


def rate_limited_transfer(payload, bandwidth_gbps, chunk_bytes=4096):
    """Send exact bytes over a local socket, pacing the sender at target Gbps."""
    if bandwidth_gbps <= 0 or chunk_bytes <= 0:
        raise ValueError("bandwidth and chunk size must be positive")
    sender, receiver = socket.socketpair()
    errors = []
    start = time.perf_counter()

    def send():
        try:
            with sender:
                sent = 0
                for offset in range(0, len(payload), chunk_bytes):
                    piece = payload[offset:offset + chunk_bytes]
                    sender.sendall(piece)
                    sent += len(piece)
                    target = start + sent * 8 / (bandwidth_gbps * 1e9)
                    remaining = target - time.perf_counter()
                    if remaining > 0:
                        time.sleep(remaining)
        except Exception as error:
            errors.append(error)

    worker = threading.Thread(target=send, daemon=True)
    worker.start()
    received = bytearray()
    with receiver:
        while True:
            piece = receiver.recv(65536)
            if not piece:
                break
            received.extend(piece)
    worker.join()
    if errors:
        raise errors[0]
    if received != payload:
        raise RuntimeError("local transport corrupted the payload")
    return bytes(received), time.perf_counter() - start


def timed_gpu(function):
    torch.cuda.synchronize()
    start = time.perf_counter()
    value = function()
    torch.cuda.synchronize()
    return value, time.perf_counter() - start


def summarize_ttft(cases):
    """Median and interpolated P95 per method, bandwidth, context length."""
    groups = defaultdict(list)
    for case in cases:
        for row in case["measurements"]:
            key = f"{row['method']}@{row['bandwidth_gbps']:g}Gbps@{case['context_tokens']}tokens"
            groups[key].append(row["ttft_wall_seconds"])
    result = {}
    for key, values in groups.items():
        ordered = sorted(values)
        rank = (len(ordered) - 1) * 0.95
        lower = int(rank)
        upper = min(lower + 1, len(ordered) - 1)
        p95 = ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)
        result[key] = {"samples": len(ordered), "median_seconds": round(median(ordered), 6),
                       "p95_seconds": round(p95, 6)}
    return result


@torch.inference_mode()
def first_token(model, context_ids, question_ids, cache=None):
    ids = question_ids if cache is not None else context_ids + question_ids
    inputs = torch.tensor([ids], dtype=torch.long, device="cuda")
    if cache is None:
        logits = model(input_ids=inputs, use_cache=False).logits
    else:
        logits = model(input_ids=inputs, past_key_values=blob_to_legacy(cache), use_cache=True).logits
    return int(logits[0, -1].argmax().item())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--lengths", type=int, nargs="+", default=[512, 1024, 1536])
    parser.add_argument("--bandwidth-gbps", type=float, nargs="+", default=[0.1, 1.0, 10.0])
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    if args.samples < 1 or min(args.lengths) < 1 or min(args.bandwidth_gbps) <= 0:
        parser.error("samples, lengths and bandwidths must be positive")
    model, tokenizer = load_local_model()
    question_ids = tokenizer.encode(QUESTION, add_special_tokens=False)
    if max(args.lengths) + len(question_ids) > model.config.max_position_embeddings:
        parser.error("context plus question exceeds model window")
    cases = load_longchat_segments(ROOT / "test_data" / "longchat.jsonl", tokenizer,
                                   lengths=args.lengths, sample_count=args.samples, continuation_tokens=2)
    data = {"model": MODEL_ID, "measurement": "single-host rate-limited socket; offline encoding excluded from TTFT",
            "parameters": {"samples": args.samples, "lengths": args.lengths,
                           "bandwidth_gbps": args.bandwidth_gbps}, "cases": []}
    save_results(args.results, data)
    for case in cases:
        context_ids = case["context_ids"]
        text = tokenizer.decode(context_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if tokenizer.encode(text, add_special_tokens=False) != context_ids:
            raise RuntimeError(f"text token roundtrip differs for {case['case_id']}; cannot compare identical contexts")
        blob = prefill(model, context_ids)
        uniform_payload, uniform_encode_s = timed_gpu(lambda: pack_uniform(blob, bits=8))
        encoder, decoder = make_codec(len(context_ids), level=2)
        cachegen_payload, cachegen_encode_s = timed_gpu(lambda: encoder.to_bytes(blob))
        payloads = {
            "text": (text.encode("utf-8"), 0.0),
            "uniform8": (uniform_payload, uniform_encode_s),
            "cachegen2": (cachegen_payload, cachegen_encode_s),
        }
        row = {"case_id": case["case_id"], "source_id": case["source_id"],
               "context_tokens": len(context_ids), "measurements": []}
        for bandwidth in args.bandwidth_gbps:
            for method, (payload, offline_encode_s) in payloads.items():
                start = time.perf_counter()
                received, transfer_s = rate_limited_transfer(payload, bandwidth)
                if method == "text":
                    text_ids = tokenizer.encode(received.decode("utf-8"), add_special_tokens=False)
                    if text_ids != context_ids:
                        raise RuntimeError("text changed after transport")
                    restored = None
                    reconstruction_s = 0.0
                    _, first_s = timed_gpu(lambda: first_token(model, text_ids, question_ids))
                else:
                    restore = (lambda: unpack_uniform(received)) if method == "uniform8" else (lambda: decoder.from_bytes(received))
                    restored, reconstruction_s = timed_gpu(restore)
                    token, first_s = timed_gpu(lambda: first_token(model, context_ids, question_ids, restored))
                    del restored, token
                wall_s = time.perf_counter() - start
                row["measurements"].append({
                    "method": method, "bandwidth_gbps": bandwidth, "bytes": len(payload),
                    "offline_encode_seconds": round(offline_encode_s, 6),
                    "transfer_seconds": round(transfer_s, 6),
                    "reconstruct_seconds": round(reconstruction_s, 6),
                    "first_token_seconds": round(first_s, 6),
                    "ttft_wall_seconds": round(wall_s, 6),
                })
        data["cases"].append(row)
        save_results(args.results, data)
        print(f"ttft {row['case_id']}", flush=True)
        del blob, uniform_payload, cachegen_payload, encoder, decoder
        torch.cuda.empty_cache()
    data["summary"] = summarize_ttft(data["cases"])
    data["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
    save_results(args.results, data)
    print(f"Results: {args.results}", flush=True)


if __name__ == "__main__":
    main()
