"""Server-only calibration and compression evaluation; --help/plan are stdlib-only."""
import argparse
import hashlib
import io
import json
from pathlib import Path
from datetime import datetime, timezone
import os
import importlib.metadata
import statistics
from paper_compression.config import CodecConfig, fingerprint
from paper_compression.provenance import provenance

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(8 << 20), b''):
            h.update(data)
    return h.hexdigest()


def load_config(path):
    value = json.loads(Path(path).read_text())
    if value.get('protocol') != 'paper-compression-v1':
        raise ValueError('Use a separate paper-compression-v1 configuration')
    codec = CodecConfig.from_dict(value['codec'])
    if codec.layers != value['model']['layers']:
        raise ValueError('Quantizer/model layer counts differ')
    train, test = value['calibration_source_ids'], value['evaluation_source_ids']
    if not train or not test or len(set(train)) != len(train) or len(set(test)) != len(test):
        raise ValueError('Provide distinct calibration and evaluation IDs')
    if set(train) & set(test):
        raise ValueError('Calibration/evaluation source leakage')
    if any(type(i) is not int or i < 0 for i in train + test):
        raise ValueError('Source IDs must be nonnegative integers')
    limit = value['model']['context_limit']
    if (type(value['calibration_tokens']) is not int or not 1 <= value['calibration_tokens'] <= limit
            or type(value['continuation_tokens']) is not int or value['continuation_tokens'] < 2):
        raise ValueError('Invalid calibration or continuation length')
    lengths = value['evaluation_lengths']
    if not lengths or len(set(lengths)) != len(lengths) or any(type(n) is not int or n < 1
            or n + value['continuation_tokens'] > limit for n in lengths):
        raise ValueError('Invalid evaluation lengths')
    if value['parameter_status'] != codec.parameter_status:
        raise ValueError('Parameter status must disclose the unresolved interpretation')
    return value, codec


def save(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def append(path, value):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def setup(cfg):
    import torch
    from experiments.engine import Engine
    if not torch.cuda.is_available():
        raise RuntimeError('Mistral calibration/evaluation must run on the CUDA server')
    torch.manual_seed(cfg['seed'])
    torch.set_num_threads(cfg['cpu_threads'])
    torch.backends.cuda.matmul.allow_tf32 = False
    engine = Engine(cfg['model'], load=False, execution=cfg['execution'])
    weights = sorted(engine.snapshot.glob('*.safetensors'))
    if not weights:
        raise ValueError('Pinned offline model weights are missing')
    names = weights + [engine.snapshot / n for n in
                       ('config.json', 'tokenizer.json', 'tokenizer_config.json')]
    model_identity = {'model': cfg['model'], 'execution': cfg['execution'],
                      'model_files_sha256': {p.name: sha256(p) for p in names},
                      'packages': {n: importlib.metadata.version(n) for n in
                                   ('torch', 'transformers', 'numpy')}}
    if cfg['model']['weight_loading'] == 'fastchat8':
        model_identity['packages']['fschat'] = importlib.metadata.version('fschat')
    return engine, model_identity


def select(engine, cfg, wanted, length, continuation=0):
    dataset = ROOT / cfg['dataset']
    found = {}
    with dataset.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            source = row['test_id']
            if source not in wanted:
                continue
            if source in found:
                raise ValueError('Duplicate source ID in dataset')
            ids = engine.tokenizer.encode(row['prompt'], add_special_tokens=False, verbose=False)
            if len(ids) < length + continuation:
                raise ValueError(f'Source {source} is too short; do not pad or silently substitute')
            found[source] = {'source_id': source, 'context_ids': ids[:length],
                             'continuation_ids': ids[length:length + continuation]}
    if set(found) != set(wanted):
        raise ValueError('Missing requested source IDs')
    return [found[source] for source in wanted]


def run_fit(args, cfg, codec_cfg):
    import torch
    from paper_compression.quantization import quantize
    from paper_compression.probability import ProfileBuilder
    if args.profile.exists():
        raise FileExistsError(args.profile)
    engine, model_identity = setup(cfg)
    dataset_hash = sha256(ROOT / cfg['dataset'])
    selected = select(engine, cfg, cfg['calibration_source_ids'], cfg['calibration_tokens'])
    builder = ProfileBuilder(codec_cfg, model_identity, dataset_hash)
    # Reserve metadata before model load; never overwrite an existing calibration run.
    work = args.profile.with_suffix(args.profile.suffix + '.calibration')
    work.mkdir(parents=True, exist_ok=False)
    save(work / 'config.json', cfg)
    source_hashes = provenance()
    save(work / 'source_hashes.json', source_hashes)
    save(work / 'status.json', {'status': 'running', 'completed_sources': 0})
    try:
        engine.load()
        for index, case in enumerate(selected):
            print('Calibrating source', case['source_id'], flush=True)
            blob = engine.prefill(case['context_ids'])
            q = quantize(blob, codec_cfg)
            builder.add(q, case['source_id'], fingerprint(case['context_ids']))
            append(work / 'sources.jsonl', {'source_id': case['source_id'],
                                          'token_sha256': fingerprint(case['context_ids'])})
            del q, blob
            torch.cuda.empty_cache()
            save(work / 'status.json', {'status': 'running', 'completed_sources': index + 1})
        if sha256(ROOT / cfg['dataset']) != dataset_hash:
            raise ValueError('Calibration dataset changed during the run')
        if provenance() != source_hashes:
            raise ValueError('Source code changed during calibration')
        profile = builder.finish()
        profile.save(args.profile)
        save(work / 'status.json', {'status': 'completed', 'profile_identity': profile.identity,
                                   'profile_bytes': args.profile.stat().st_size,
                                   'completed_sources': len(selected)})
        print('PROFILE COMPLETE:', args.profile, flush=True)
    except BaseException as exc:
        save(work / 'status.json', {'status': 'failed', 'error_type': type(exc).__name__, 'error': str(exc)})
        raise


def run_evaluate(args, cfg, codec_cfg):
    import torch
    from paper_compression.probability import ProbabilityProfile
    from paper_compression.codec import PaperCodec
    from paper_compression.quantization import quantize, reconstruct
    profile = ProbabilityProfile.load(args.profile)
    profile_hash = sha256(args.profile)
    if profile.config != codec_cfg:
        raise ValueError('Profile quantizer differs from evaluation configuration')
    dataset_hash = sha256(ROOT / cfg['dataset'])
    for source in cfg['evaluation_source_ids']:
        profile.require_held_out(source, dataset_hash)
    engine, model_identity = setup(cfg)
    profile.require_model(model_identity)
    # Select and lock every input before loading the model, so invalid data fails early.
    selected = [case for length in cfg['evaluation_lengths'] for case in select(
        engine, cfg, cfg['evaluation_source_ids'], length, cfg['continuation_tokens'])]
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / 'config.json', cfg)
    source_hashes = provenance()
    save(args.output / 'source_hashes.json', source_hashes)
    save(args.output / 'model_identity.json', model_identity)
    save(args.output / 'selection.json', selected)
    save(args.output / 'runtime.json', {'gpu': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda,
                                     'profile_identity': profile.identity, 'backend': args.backend,
                                     'profile_file_sha256': profile_hash,
                                     'parameter_status': codec_cfg.parameter_status,
                                     'started_utc': datetime.now(timezone.utc).isoformat()})
    rows = []
    state = {'status': 'running', 'completed_cases': 0, 'expected_cases': len(selected)}
    save(args.output / 'status.json', state)
    try:
        engine.load()
        codec = PaperCodec(profile, model_identity, args.backend)
        for case in selected:
            context, continuation = case['context_ids'], case['continuation_ids']
            print('Evaluating', case['source_id'], 'tokens', len(context), flush=True)
            blob = engine.prefill(context)
            raw_score = engine.nll(blob, context, continuation)
            q = quantize(blob, codec_cfg)
            direct = reconstruct(q)
            direct_score = engine.nll(direct, context, continuation)
            payload, byte_info = codec.encode_quantized(q)
            decoded_q = codec.decode_quantized(payload, 'cuda')
            for key in ('anchor', 'delta', 'anchor_scales'):
                if not torch.equal(getattr(q, key), getattr(decoded_q, key)):
                    raise ValueError('Entropy roundtrip changed quantized data: ' + key)
            if q.delta_scales is not None and not torch.equal(q.delta_scales, decoded_q.delta_scales):
                raise ValueError('Entropy roundtrip changed delta scales')
            restored = reconstruct(decoded_q)
            if not torch.equal(direct, restored):
                raise ValueError('Direct and entropy-restored KV differ')
            restored_score = engine.nll(restored, context, continuation)
            if direct_score['token_nll'] != restored_score['token_nll']:
                raise ValueError('Same restored KV produced different NLL; inspect inference determinism')
            raw_container = io.BytesIO()
            torch.save(blob.cpu(), raw_container)
            row = {'source_id': case['source_id'], 'context_tokens': len(context),
                   'raw': raw_score, 'quantize_only': direct_score, 'roundtrip': restored_score,
                   'delta_nll': restored_score['nll'] - raw_score['nll'],
                   'raw_serialized_bytes': raw_container.tell(), **byte_info,
                   'compression_ratio': raw_container.tell() / len(payload),
                   'quantized_symbols_exact': True, 'restored_kv_exact': True,
                   'packet_sha256': hashlib.sha256(payload).hexdigest()}
            append(args.output / 'records.jsonl', row)
            rows.append(row)
            state['completed_cases'] = len(rows)
            save(args.output / 'status.json', state)
            del blob, q, direct, decoded_q, restored, payload, raw_container
            torch.cuda.empty_cache()
        summary = []
        for length in cfg['evaluation_lengths']:
            group = [r for r in rows if r['context_tokens'] == length]
            profile_share = args.profile.stat().st_size / len(group)
            summary.append({'context_tokens': length, 'independent_sources': len(group),
                            'mean_delta_nll': statistics.mean(r['delta_nll'] for r in group),
                            'median_delta_nll': statistics.median(r['delta_nll'] for r in group),
                            'raw_bytes_over_packet_bytes': sum(r['raw_serialized_bytes'] for r in group) /
                                                          sum(r['complete_bytes'] for r in group),
                            'profile_bytes': args.profile.stat().st_size,
                            'profile_share_bytes_per_packet_at_this_length': profile_share,
                            'raw_bytes_over_packet_plus_profile_share': sum(r['raw_serialized_bytes'] for r in group) /
                                (sum(r['complete_bytes'] for r in group) + args.profile.stat().st_size),
                            'total_escape_count': sum(r['escape_count'] for r in group)})
        if sha256(ROOT / cfg['dataset']) != dataset_hash:
            raise ValueError('Evaluation data changed during the run')
        if provenance() != source_hashes or sha256(args.profile) != profile_hash:
            raise ValueError('Source code or probability profile changed during evaluation')
        save(args.output / 'summary.json', {'parameter_status': codec_cfg.parameter_status,
                                          'quality_metric': 'held-out continuation NLL; not paper QA equivalence',
                                          'by_length': summary})
        state['status'] = 'completed'
        save(args.output / 'status.json', state)
        print('EVALUATION COMPLETE:', args.output, flush=True)
    except BaseException as exc:
        state.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        save(args.output / 'status.json', state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'fit', 'evaluate'))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--profile', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--backend', choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args()
    cfg, codec = load_config(args.config)
    if args.action == 'plan':
        print(json.dumps({'protocol': cfg['protocol'], 'codec': codec.to_dict(),
                          'layer_boundaries': codec.boundaries,
                          'calibration_sources': cfg['calibration_source_ids'],
                          'evaluation_sources': cfg['evaluation_source_ids'],
                          'model_weight_loading': cfg['model']['weight_loading'],
                          'local_execution_validated': False}, indent=2))
        return
    if args.profile is None or (args.action == 'evaluate' and args.output is None):
        parser.error('fit requires --profile; evaluate requires --profile and --output')
    if args.action == 'fit':
        run_fit(args, cfg, codec)
    else:
        run_evaluate(args, cfg, codec)


if __name__ == '__main__':
    main()
