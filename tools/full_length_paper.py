"""Server-only full-length run of the paper's encoder (section 5.2) and its Fig. 15 steps.

  paper-r*    change-based encoding, layer-wise bins 0.5/1/1.5, offline per-layer/channel tables.
  nolayer-r*  the same without layer-wise bins (all 1): "Quant + AC + Change" in Fig. 15.
  nodelta-r*  the same without change-based encoding: the delta's own control.
  *-abs       the literal reading: bins 0.5/1/1.5 in raw KV units, no per-vector scaling.
Tables come from IDs 30-49 (conversation only, as no query exists offline) and are not in the
packet. The range r is the disclosed interpretation of the paper's unspecified bin unit.
Evaluation uses IDs 0-19. Nothing executes at import time.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

from paper_compression.config import CodecConfig
from paper_compression.probability import ProbabilityProfile, ProfileBuilder, grouped_histogram
from tools.full_length_qa import GENERATION_TOKENS, ROOT, done_keys, load_rows, score, split_prompt

RANGES = (2., 3., 5., 7.5)


def configs(layers):
    result = {}
    for r in RANGES:
        result[f'paper-r{r:g}'] = CodecConfig(layers=layers, vector_range=r)
    for r in (3., 5.):
        result[f'nolayer-r{r:g}'] = CodecConfig(layers=layers, vector_range=r, delta_bin_widths=(1., 1., 1.))
    for r in (3., 5., 7.5):
        result[f'nodelta-r{r:g}'] = CodecConfig(layers=layers, vector_range=r, use_delta=False)
    result['paper-r4'] = CodecConfig(layers=layers, vector_range=4.)
    result['nolayer-r4'] = CodecConfig(layers=layers, vector_range=4., delta_bin_widths=(1., 1., 1.))
    result['nodelta-r4'] = CodecConfig(layers=layers, vector_range=4., use_delta=False)
    result['paper-abs'] = CodecConfig(layers=layers, delta_scale_mode='absolute')
    result['nolayer-abs'] = CodecConfig(layers=layers, delta_scale_mode='absolute', delta_bin_widths=(1., 1., 1.))
    result['nodelta-abs'] = CodecConfig(layers=layers, delta_scale_mode='absolute', use_delta=False)
    return result


def selected(layers, names):
    every = configs(layers)
    if names is None:
        return every
    unknown = set(names) - set(every)
    if unknown:
        raise ValueError(f'Unknown configs: {sorted(unknown)}')
    return {n: every[n] for n in names}


class HostCountBuilder(ProfileBuilder):
    """Histograms run on the GPU; the counts stay in host memory beside a full-length prefill."""
    def add(self, quantized, source_id, input_sha256):
        from paper_compression.quantization import delta_symbols
        source_id = str(source_id)
        if source_id in self.sources or quantized.config != self.config:
            raise ValueError('Duplicate calibration source or changed quantizer')
        layers, _, heads, _, dim = quantized.shape
        self.geometry = {'layers': layers, 'heads': heads, 'head_dim': dim}
        delta, _ = delta_symbols(quantized.delta, self.config)
        for kind, symbols, alphabet in (('anchor', quantized.anchor, 256), ('delta', delta, self.config.delta_alphabet)):
            hist = grouped_histogram(symbols, 'layer_channel', alphabet).cpu()
            self.counts = self.counts or {}
            self.counts[kind] = self.counts[kind] + hist if kind in self.counts else hist
        self.sources[source_id] = input_sha256


def prepare(args):
    import torch
    from experiments.engine import Engine
    cfg = json.loads((ROOT / 'paper_compression/configs/core-study.json').read_text())
    if not torch.cuda.is_available():
        raise RuntimeError('This experiment must run on the CUDA server')
    torch.manual_seed(cfg['seed'])
    torch.set_num_threads(cfg['cpu_threads'])
    torch.backends.cuda.matmul.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=True)
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    data_hash = digest(ROOT / 'test_data/longchat.jsonl')
    meta = {'commit': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
            'tool_sha256': digest(__file__), 'qa_tool_sha256': digest(ROOT / 'tools/full_length_qa.py'),
            'dataset_sha256': data_hash, 'model': cfg['model'], 'execution': cfg['execution'],
            'generation_tokens': GENERATION_TOKENS, 'calibration_ids': args.calibration_ids,
            'test_ids': args.test_ids, 'arms': args.arms, 'ranges': RANGES, 'configs': args.configs, 'torch': torch.__version__}
    (args.output / 'run_meta.json').write_text(json.dumps(meta, indent=1))
    engine = Engine(cfg['model'], load=True, execution=cfg['execution'])
    return torch, engine, cfg, data_hash


def calibrate(args, torch, engine, cfg, data_hash):
    from paper_compression.quantization import quantize
    folder = args.output / 'profiles'
    folder.mkdir(exist_ok=True)
    pending = {n: c for n, c in selected(cfg['model']['layers'], args.configs).items() if not (folder / f'{n}.pt').exists()}
    if not pending:
        return
    builders = {n: HostCountBuilder(c, cfg['model']['revision'], data_hash) for n, c in pending.items()}
    rows = load_rows(args.calibration_ids)
    start = time.monotonic()
    for source in args.calibration_ids:
        ids = split_prompt(engine.tokenizer, rows[source]['prompt'])['clean_question'][0]
        blob = engine.prefill(ids).cpu()  # quantize on the host: the model leaves too little GPU memory
        torch.cuda.empty_cache()
        checksum = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        for name, builder in builders.items():
            q = quantize(blob, builder.config)
            builder.add(q, source, checksum)
            del q
            torch.cuda.empty_cache()
        del blob
        torch.cuda.empty_cache()
        print('CALIBRATED', source, f'{time.monotonic() - start:.0f}s', flush=True)
    for name, builder in builders.items():
        builder.finish().save(folder / f'{name}.pt')


def evaluate(args, torch, engine, cfg, data_hash):
    from experiments.engine import validate_blob
    from paper_compression.codec import PaperCodec
    from paper_compression.quantization import quantize, reconstruct
    identity = cfg['model']['revision']
    profiles = {n: ProbabilityProfile.load(args.output / 'profiles' / f'{n}.pt')
                for n in selected(cfg['model']['layers'], args.configs)}
    records = args.output / 'records.jsonl'
    done = done_keys(records)
    rows = load_rows(args.test_ids)
    methods = ('raw', *profiles)
    start = time.monotonic()
    for source in args.test_ids:
        row = rows[source]
        arms = split_prompt(engine.tokenizer, row['prompt'])
        for arm in args.arms:
            if all((source, arm, m) in done for m in methods):
                continue
            kv_ids, suffix = arms[arm]
            torch.cuda.reset_peak_memory_stats()
            blob = engine.prefill(kv_ids)
            torch.cuda.empty_cache()
            raw_prediction = engine.generate(suffix, blob, GENERATION_TOKENS)['prediction']
            host_blob = blob.cpu()
            for method in methods:
                if (source, arm, method) in done:
                    continue
                t0 = time.monotonic()
                record = {'source_id': source, 'arm': arm, 'method': method, 'kv_tokens': len(kv_ids),
                          'suffix_tokens': len(suffix), 'answer': row['label'][0]}
                if method == 'raw':
                    restored = blob
                    record.update(bytes=blob.numel() * 2, mse=0.)
                else:
                    profile = profiles[method]
                    profile.require_held_out(source, data_hash)
                    codec = PaperCodec(profile, identity, 'cuda')
                    q = quantize(host_blob, profile.config)
                    payload, stats = codec.encode_quantized(q)
                    del q
                    torch.cuda.empty_cache()
                    restored = reconstruct(codec.decode_quantized(payload, 'cpu')).to('cuda')
                    validate_blob(restored, blob)
                    mse = sum(float(((restored[i].float() - blob[i].float()) ** 2).mean())
                              for i in range(blob.shape[0])) / blob.shape[0]
                    record.update(bytes=len(payload), mse=mse, stream_bytes=stats['arithmetic_stream_bytes'],
                                  escape_count=stats['escape_count'], escape_bytes=stats['escape_bytes'],
                                  shared_table_bytes=(args.output / 'profiles' / f'{method}.pt').stat().st_size)
                    del payload, codec
                generated = engine.generate(suffix, restored, GENERATION_TOKENS)
                record.update(prediction=generated['prediction'], stop_reason=generated['stop_reason'],
                              identical_to_raw=generated['prediction'] == raw_prediction,
                              **score(generated['prediction'], row['label']),
                              method_seconds=time.monotonic() - t0, peak_gpu_bytes=torch.cuda.max_memory_allocated())
                with records.open('a') as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                print(source, arm, method, record['bytes'], int(record['includes_answer']),
                      repr(generated['prediction'][:60]), f"{record['method_seconds']:.1f}s", flush=True)
                if method != 'raw':
                    del restored
                torch.cuda.empty_cache()
            del blob, host_blob
            torch.cuda.empty_cache()
    (args.output / 'status.json').write_text(json.dumps(
        {'status': 'completed', 'wall_seconds': time.monotonic() - start, 'test_ids': args.test_ids}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--calibration-ids', type=int, nargs='+', required=True)
    parser.add_argument('--test-ids', type=int, nargs='+', required=True)
    parser.add_argument('--arms', nargs='+', default=['clean_question'], choices=['clean_question', 'author'])
    parser.add_argument('--configs', nargs='+', help='Subset of configs (default: all)')
    args = parser.parse_args()
    if set(args.calibration_ids) & set(args.test_ids):
        parser.error('Calibration and test documents must not overlap')
    torch, engine, cfg, data_hash = prepare(args)
    calibrate(args, torch, engine, cfg, data_hash)
    evaluate(args, torch, engine, cfg, data_hash)


if __name__ == '__main__':
    main()
