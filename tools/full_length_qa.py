"""Server-only full-length LongChat quality check: baselines vs. the author's codec.

Follows the author's main.py / run_cachegen.py:
  arm "author":         KV covers the whole prompt except its last token (the question is compressed).
  arm "clean_question": KV covers the conversation only; the question stays uncompressed (diagnostic).
No calibration or tuning happens here. Nothing executes at import time.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
QUESTION_MARKER = ' Now the record ends.'
GENERATION_TOKENS = 20  # max_new_tokens in the author's run_cachegen.py
METHODS = ('raw', 'uniform8', 'uniform4', 'uniform3', 'cachegen1', 'cachegen2', 'cachegen3')


def pack_uniform3(blob):
    """Same scales and container as pack_uniform, with 3-bit packed symbols (8 values -> 3 bytes)."""
    import torch
    layers, kinds, heads, tokens, width = blob.shape
    groups = blob.permute(0, 1, 3, 2, 4).reshape(layers, kinds, tokens, heads * width)
    maxima = groups.abs().amax(dim=-1, keepdim=True)
    safe = maxima.clamp_min(torch.finfo(maxima.dtype).tiny)
    quantized = torch.round(groups.float() / safe.float() * 3).clamp(-3, 3).to(torch.int8).flatten()
    symbols = (quantized + 3).cpu().to(torch.int64)  # 8 bytes per value: stay off the GPU
    count = symbols.numel()
    pad = -count % 8
    if pad:
        symbols = torch.cat((symbols, torch.zeros(pad, device=symbols.device, dtype=torch.int64)))
    parts = []
    for start in range(0, symbols.numel(), 8 << 24):
        block = symbols[start:start + (8 << 24)].reshape(-1, 8)
        word = sum(block[:, i] << (3 * i) for i in range(8))
        parts.append(torch.stack([(word >> s) & 255 for s in (0, 8, 16)], dim=1).to(torch.uint8).flatten())
    stream = io.BytesIO()
    torch.save({'bits': 3, 'shape': tuple(blob.shape), 'count': count,
                'values': torch.cat(parts), 'maxima': maxima.cpu()}, stream)
    return stream.getvalue()


def unpack_uniform3(payload, device='cuda'):
    import torch
    saved = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
    if saved['bits'] != 3:
        raise ValueError('Expected a 3-bit payload')
    triples = saved['values'].reshape(-1, 3).to(torch.int64)
    word = triples[:, 0] | (triples[:, 1] << 8) | (triples[:, 2] << 16)
    symbols = torch.stack([(word >> (3 * i)) & 7 for i in range(8)], dim=1).flatten()[:saved['count']]
    if symbols.numel() and int(symbols.max()) > 6:
        raise ValueError('Invalid 3-bit symbol')
    layers, kinds, heads, tokens, width = saved['shape']
    maxima = saved['maxima'].to(device)
    groups = (symbols - 3).to(torch.int8).reshape(layers, kinds, tokens, heads * width).to(device).to(maxima.dtype)
    restored = groups / 3 * maxima
    return restored.reshape(layers, kinds, tokens, heads, width).permute(0, 1, 3, 2, 4)


def words(text):
    import re
    return ' '.join(re.findall(r'\w+', text.casefold()))


def includes(prediction, answer):
    """The paper's definition: the generated answer includes the ground-truth topic."""
    return f' {words(answer)} ' in f' {words(prediction)} '


def score(prediction, labels):
    first_line = prediction.lstrip('\n').split('\n', 1)[0]
    others = [t for t in labels[1:] if includes(first_line, t)]
    return {'includes_answer': includes(prediction, labels[0]),
            'includes_answer_first_line': includes(first_line, labels[0]),
            'first_line_names_other_topic': bool(others)}


def load_rows(ids):
    rows = {}
    with (ROOT / 'test_data/longchat.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if row['test_id'] in ids:
                    rows[row['test_id']] = row
    if set(rows) != set(ids):
        raise ValueError('Missing LongChat rows')
    return rows


def split_prompt(tokenizer, prompt):
    """Return (kv_ids, suffix_ids) for both arms; tokenized like the author (with BOS)."""
    ids = tokenizer.encode(prompt)
    if ids[0] != tokenizer.bos_token_id:
        raise ValueError('Expected the author-style BOS token')
    if prompt.count(QUESTION_MARKER) != 1:
        raise ValueError('Question marker must occur exactly once')
    cut = prompt.index(QUESTION_MARKER)
    head = tokenizer.encode(prompt[:cut])
    tail = tokenizer.encode(prompt[cut:], add_special_tokens=False)
    return {'author': (ids[:-1], ids[-1:]), 'clean_question': (head, tail)}


def encode_method(engine, blob, method):
    from experiments.engine import timed
    if method == 'uniform3':
        payload, seconds = timed(lambda: pack_uniform3(blob))
        return payload, seconds, unpack_uniform3
    return engine.encode(blob, method)


def done_keys(path):
    if not path.exists():
        return set()
    with path.open() as stream:
        return {(r['source_id'], r['arm'], r['method']) for r in map(json.loads, stream)}


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
    meta = {'commit': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
            'tool_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'dataset_sha256': hashlib.sha256((ROOT / 'test_data/longchat.jsonl').read_bytes()).hexdigest(),
            'model': cfg['model'], 'execution': cfg['execution'], 'generation_tokens': GENERATION_TOKENS,
            'source_ids': args.ids, 'arms': args.arms, 'methods': list(METHODS),
            'torch': torch.__version__}
    (args.output / 'run_meta.json').write_text(json.dumps(meta, indent=1))
    records = args.output / 'records.jsonl'
    done = done_keys(records)
    engine = Engine(cfg['model'], load=True, execution=cfg['execution'])
    rows = load_rows(args.ids)
    start = time.monotonic()
    for source in args.ids:
        row = rows[source]
        arms = split_prompt(engine.tokenizer, row['prompt'])
        for arm in args.arms:
            if all((source, arm, m) in done for m in METHODS):
                continue
            kv_ids, suffix = arms[arm]
            torch.cuda.reset_peak_memory_stats()
            blob = engine.prefill(kv_ids)
            raw_prediction = None
            for method in METHODS:
                if (source, arm, method) in done and method != 'raw':
                    continue
                t0 = time.monotonic()
                payload, encode_seconds, restore = encode_method(engine, blob, method)
                restored = restore(payload)
                validate_blob(restored, blob)
                mse = sum(float(((restored[i].float() - blob[i].float()) ** 2).mean())
                          for i in range(blob.shape[0])) / blob.shape[0]
                generated = engine.generate(suffix, restored, GENERATION_TOKENS)
                if method == 'raw':
                    raw_prediction = generated['prediction']
                record = {'source_id': source, 'arm': arm, 'method': method, 'kv_tokens': len(kv_ids),
                          'suffix_tokens': len(suffix), 'bytes': len(payload), 'mse': mse,
                          'prediction': generated['prediction'], 'stop_reason': generated['stop_reason'],
                          'identical_to_raw': generated['prediction'] == raw_prediction,
                          **score(generated['prediction'], row['label']), 'answer': row['label'][0],
                          'encode_seconds': encode_seconds, 'method_seconds': time.monotonic() - t0,
                          'peak_gpu_bytes': torch.cuda.max_memory_allocated()}
                if (source, arm, method) not in done:
                    with records.open('a') as stream:
                        stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                print(source, arm, method, len(payload), int(record['includes_answer']),
                      repr(generated['prediction'][:70]), f"{record['method_seconds']:.1f}s", flush=True)
                del restored, payload
                torch.cuda.empty_cache()
            del blob
            torch.cuda.empty_cache()
    (args.output / 'status.json').write_text(json.dumps(
        {'status': 'completed', 'wall_seconds': time.monotonic() - start, 'source_ids': args.ids}))


def self_test():
    """CPU check of the 3-bit wire format; no model involved."""
    import torch
    for shape in ((2, 2, 3, 5, 4), (3, 2, 2, 7, 3)):
        blob = torch.randn(shape).half()
        restored = unpack_uniform3(pack_uniform3(blob), device='cpu')
        groups = blob.permute(0, 1, 3, 2, 4).reshape(shape[0], 2, shape[3], shape[2] * shape[4]).float()
        scale = groups.abs().amax(-1, keepdim=True)
        expected = (torch.round(groups / scale * 3).clamp(-3, 3) / 3 * scale)
        actual = restored.permute(0, 1, 3, 2, 4).reshape(groups.shape).float()
        if restored.dtype != blob.dtype or not torch.allclose(actual, expected, atol=2e-3):
            raise AssertionError('3-bit roundtrip mismatch')
    if not includes('The first topic was the role of art in society.', 'The role of art in society'):
        raise AssertionError('includes() failed on a real answer')
    if includes('The role of artists', 'The role of art'):
        raise AssertionError('includes() must respect word boundaries')
    print('self-test ok')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--ids', type=int, nargs='+')
    parser.add_argument('--arms', nargs='+', default=['author', 'clean_question'],
                        choices=['author', 'clean_question'])
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if args.output is None or not args.ids:
        parser.error('--output and --ids are required')
    return run(args)


if __name__ == '__main__':
    main()
