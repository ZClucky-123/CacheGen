"""Server-only full-length timing for time-to-first-token (TTFT), paper section 7.3 / Fig. 8.

For each document (KV covers the conversation only; the question is computed fresh):
  text      time to prefill conversation + question from scratch and get the first token;
  methods   bytes, decode time (payload -> KV on the GPU) and the time to compute the
            question on the restored KV and get the first token.
Transfer time is not simulated here: analyze.py adds bytes*8/bandwidth afterwards.
Every timing is the median of REPEATS runs after one warm-up, with CUDA synchronized.
Nothing executes at import time.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time

from tools.full_length_qa import ROOT, done_keys, load_rows, split_prompt

METHODS = ('raw', 'uniform8', 'uniform4', 'cachegen3', 'cachegen2', 'cachegen1', 'ac_b12')
REPEATS = 3


def median_time(torch, fn):
    value = fn()  # warm-up
    seconds = []
    for _ in range(REPEATS):
        value = None  # never hold two restored copies at once
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        start = time.perf_counter()
        value = fn()
        torch.cuda.synchronize()
        seconds.append(time.perf_counter() - start)
    return value, statistics.median(seconds), seconds


def encode(engine, blob, method):
    from tools.full_length_ac import SCHEDULES, encode_method
    if method in SCHEDULES:
        payload, _, restore = encode_method(engine, blob, method)
    else:
        payload, _, restore = engine.encode(blob, method)
    return payload, restore


def run(args):
    import torch
    from experiments.engine import Engine, validate_blob
    cfg = json.loads((ROOT / 'paper_compression/configs/core-study.json').read_text())
    if not torch.cuda.is_available():
        raise RuntimeError('This experiment must run on the CUDA server')
    torch.manual_seed(cfg['seed'])
    torch.set_num_threads(cfg['cpu_threads'])
    torch.backends.cuda.matmul.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=True)
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    meta = {'commit': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
            'tool_sha256': digest(__file__), 'qa_tool_sha256': digest(ROOT / 'tools/full_length_qa.py'),
            'ac_tool_sha256': digest(ROOT / 'tools/full_length_ac.py'),
            'dataset_sha256': digest(ROOT / 'test_data/longchat.jsonl'), 'model': cfg['model'],
            'execution': cfg['execution'], 'ids': args.ids, 'methods': METHODS, 'repeats': REPEATS,
            'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__}
    (args.output / 'run_meta.json').write_text(json.dumps(meta, indent=1))
    records = args.output / 'records.jsonl'
    done = done_keys(records)
    engine = Engine(cfg['model'], load=True, execution=cfg['execution'])
    rows = load_rows(args.ids)
    start = time.monotonic()
    write = lambda rec: records.open('a').write(json.dumps(rec, ensure_ascii=False) + '\n')
    for source in args.ids:
        context, suffix = split_prompt(engine.tokenizer, rows[source]['prompt'])['clean_question']
        if all((source, 'clean_question', m) in done for m in ('text', *METHODS)):
            continue
        base = {'source_id': source, 'arm': 'clean_question', 'kv_tokens': len(context), 'suffix_tokens': len(suffix)}
        direct, text_s, text_runs = median_time(torch, lambda: engine.first(context, suffix))
        torch.cuda.empty_cache()
        if (source, 'clean_question', 'text') not in done:
            text = engine.text_payload(context)
            write({**base, 'method': 'text', 'bytes': len(text),
                   'decode_seconds': 0., 'first_token_seconds': text_s, 'first_token_runs': text_runs,
                   'first_token': direct, 'matches_direct': True})
            print(source, 'text', len(text), f'{text_s:.3f}s', flush=True)
        host = engine.prefill(context).cpu()  # the original KV stays off the GPU while decoding
        reference = torch.empty(host.shape, dtype=host.dtype, device='meta')
        torch.cuda.empty_cache()
        for method in METHODS:
            if (source, 'clean_question', method) in done:
                continue
            blob = host.to('cuda')
            payload, restore = encode(engine, blob, method)
            del blob
            torch.cuda.empty_cache()
            restored, decode_s, decode_runs = median_time(torch, lambda: restore(payload))
            validate_blob(restored, reference)
            token, first_s, first_runs = median_time(torch, lambda: engine.first(context, suffix, restored))
            write({**base, 'method': method, 'bytes': len(payload), 'decode_seconds': decode_s,
                   'decode_runs': decode_runs, 'first_token_seconds': first_s, 'first_token_runs': first_runs,
                   'first_token': token, 'matches_direct': token == direct,
                   'peak_gpu_bytes': torch.cuda.max_memory_allocated()})
            print(source, method, len(payload), f'decode {decode_s:.3f}s first {first_s:.3f}s', token == direct, flush=True)
            del restored, payload, restore
            torch.cuda.empty_cache()
        del host
        torch.cuda.empty_cache()
    (args.output / 'status.json').write_text(json.dumps(
        {'status': 'completed', 'wall_seconds': time.monotonic() - start, 'ids': args.ids}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ids', type=int, nargs='+', required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
