"""Server-only follow-up to full_length_qa: separate arithmetic coding from layer-wise precision.

Every method sends its per-context probability table in the packet, as the author's code does:
  uniform8_ac    the 8-bit baseline plus arithmetic coding (lossless on top of uniform8);
  ac_b16..ac_b6  one precision for every layer, through the author's codec;
  cachegen1, layer_a..layer_d  shallow layers finer than deep ones, through the author's codec.
Bins B map to a maximum symbol of B//2-1, so ac_b16 and ac_b8 quantize like 4-bit and 3-bit.
No calibration or tuning happens here. Nothing executes at import time.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import time

from tools.full_length_qa import GENERATION_TOKENS, ROOT, done_keys, load_rows, score, split_prompt

# name: (K first 10 layers, K layers 10-19, K layers 20-31, V first 2 layers, V remaining layers)
SCHEDULES = {
    'ac_b16': (16, 16, 16, 16, 16), 'ac_b12': (12, 12, 12, 12, 12), 'ac_b10': (10, 10, 10, 10, 10),
    'ac_b8': (8, 8, 8, 8, 8), 'ac_b6': (6, 6, 6, 6, 6),
    'cachegen1': (16, 16, 12, 16, 12), 'layer_a': (16, 12, 10, 16, 10), 'layer_b': (16, 10, 8, 16, 8),
    'layer_c': (12, 8, 8, 12, 8), 'layer_d': (12, 8, 6, 12, 6),
}
METHODS = ('raw', 'uniform8_ac', *SCHEDULES)


def codec(engine, tokens, name):
    from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
    from lmcache.storage_backend.serde.cachegen_basics import CacheGenConfig
    from lmcache.storage_backend.serde.cachegen_decoder import CacheGenDeserializer
    from lmcache.storage_backend.serde.cachegen_encoder import CacheGenSerializer
    k1, k2, k3, v1, v2 = SCHEDULES[name]
    spec = engine.spec
    schedule = CacheGenConfig(*spec['key_boundaries'], k1, k2, k3, spec['value_boundary'], v1, v2)
    config = LMCacheEngineConfig.from_defaults(chunk_size=tokens)
    metadata = LMCacheEngineMetadata(spec['id'], 1, 0, 'huggingface')
    return (CacheGenSerializer(config, metadata, cachegen_config=schedule),
            CacheGenDeserializer(config, metadata, cachegen_config=schedule))


def uniform8_ac_pack(blob):
    """pack_uniform's 8-bit symbols, coded per K/V layer and channel with a table sent in the packet."""
    import torch
    from paper_compression import entropy
    from paper_compression.probability import grouped_histogram, make_cdf
    layers, kinds, heads, tokens, width = blob.shape
    groups = blob.permute(1, 0, 3, 2, 4).reshape(kinds * layers, tokens, heads * width)
    maxima = groups.abs().amax(dim=-1, keepdim=True)
    safe = maxima.clamp_min(torch.finfo(maxima.dtype).tiny).float()
    symbols = torch.empty(groups.shape, device=groups.device, dtype=torch.uint8)
    for row in range(groups.shape[0]):
        symbols[row] = (torch.round(groups[row].float() / safe[row] * 127).clamp(-127, 127) + 127).to(torch.uint8)
    cdf = make_cdf(grouped_histogram(symbols, 'layer_channel', 256).cpu())
    stream = io.BytesIO()
    torch.save({'shape': tuple(blob.shape), 'stream': entropy.encode(symbols, cdf, 'cuda'),
                'cdf': cdf, 'maxima': maxima.cpu()}, stream)
    return stream.getvalue()


def uniform8_ac_unpack(payload):
    import torch
    from paper_compression import entropy
    saved = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
    layers, kinds, heads, tokens, width = saved['shape']
    rows = kinds * layers
    symbols = entropy.decode(saved['stream'], saved['cdf'], (rows, tokens, heads * width), 'cuda')
    maxima = saved['maxima'].to('cuda')
    groups = (symbols.to(torch.int16) - 127).to(maxima.dtype) / 127 * maxima
    return groups.reshape(kinds, layers, tokens, heads, width).permute(1, 0, 3, 2, 4)


def encode_method(engine, blob, method):
    from experiments.engine import timed
    if method == 'uniform8_ac':
        payload, seconds = timed(lambda: uniform8_ac_pack(blob))
        return payload, seconds, uniform8_ac_unpack
    if method == 'raw':
        return engine.encode(blob, 'raw')
    encoder, decoder = codec(engine, blob.shape[3], method)
    payload, seconds = timed(lambda: encoder.to_bytes(blob))
    return payload, seconds, decoder.from_bytes


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
            'dataset_sha256': digest(ROOT / 'test_data/longchat.jsonl'), 'model': cfg['model'],
            'execution': cfg['execution'], 'generation_tokens': GENERATION_TOKENS, 'source_ids': args.ids,
            'arms': args.arms, 'methods': list(METHODS), 'schedules': SCHEDULES, 'torch': torch.__version__}
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ids', type=int, nargs='+', required=True)
    parser.add_argument('--arms', nargs='+', default=['author', 'clean_question'],
                        choices=['author', 'clean_question'])
    run(parser.parse_args())


if __name__ == '__main__':
    main()
