"""Server-only, split-locked core-method study. No execution at import time."""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import statistics
import time

from paper_compression.config import fingerprint
from paper_compression.study_config import validate, candidates, select_candidates
from tools.paper_compression import ROOT, sha256, save, append, setup, select, provenance


def read(path):
    return json.loads(Path(path).read_text())


def records(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


@contextmanager
def run_folder(args, cfg):
    args.output.mkdir(parents=True, exist_ok=False)
    sources = provenance()
    data_hash = sha256(ROOT / cfg['dataset'])
    dependencies = {}
    if args.profiles is not None:
        dependencies.update({str(p): sha256(p) for p in args.profiles.glob('*.pt')})
        dependencies[str(args.profiles / 'index.json')] = sha256(args.profiles / 'index.json')
    if args.selection is not None:
        for p in (args.selection, args.selection.parent / 'records.jsonl'):
            dependencies[str(p)] = sha256(p)
    save(args.output / 'config.json', cfg)
    save(args.output / 'source_hashes.json', sources)
    save(args.output / 'dependency_hashes.json', dependencies)
    save(args.output / 'status.json', {'status': 'running', 'action': args.action})
    start = time.monotonic()
    try:
        yield data_hash
        if sources != provenance() or data_hash != sha256(ROOT / cfg['dataset']):
            raise ValueError('Source code or dataset changed during run')
        if any(sha256(p) != expected for p, expected in dependencies.items()):
            raise ValueError('Profile or frozen validation dependency changed during run')
        save(args.output / 'status.json', {'status': 'completed', 'action': args.action,
                                         'wall_seconds': time.monotonic() - start})
    except BaseException as exc:
        save(args.output / 'status.json', {'status': 'failed', 'action': args.action,
             'error': str(exc), 'error_type': type(exc).__name__, 'wall_seconds': time.monotonic() - start})
        raise


def inputs(engine, cfg, split, lengths=None):
    return [case for length in (lengths or cfg['lengths']) for case in select(
        engine, cfg, cfg['splits'][split], length, cfg['continuation_tokens'])]


def load_profiles(folder, cfg, model_identity, data_hash, names=None):
    from paper_compression.probability import ProbabilityProfile
    index = read(folder / 'index.json')
    if (index['study_identity'] != fingerprint(cfg) or index['dataset_sha256'] != data_hash
            or index['source_identity'] != fingerprint(provenance())):
        raise ValueError('Calibration index belongs to another study/data file')
    result = {}
    options = candidates(cfg)
    for name in (names or options):
        path = folder / (name + '.pt')
        if sha256(path) != index['profiles'][name]['sha256']:
            raise ValueError('Profile file changed: ' + name)
        p = ProbabilityProfile.load(path)
        p.require_model(model_identity)
        if p.config != options[name] or p.identity != index['profiles'][name]['identity']:
            raise ValueError('Profile quantizer/content differs: ' + name)
        if p.metadata['calibration_sources'] != sorted(map(str, cfg['splits']['calibration'])):
            raise ValueError('Profile uses wrong calibration sources')
        result[name] = p
    return result


def calibrate(args, cfg):
    import torch
    from paper_compression.probability import GPUProfileBuilder
    from paper_compression.quantization import quantize
    with run_folder(args, cfg) as data_hash:
        engine, identity = setup(cfg)
        selected = inputs(engine, cfg, 'calibration', [cfg['calibration_tokens']])
        save(args.output / 'model_identity.json', identity)
        save(args.output / 'selection.json', selected)
        engine.load()
        index = {'study_identity': fingerprint(cfg), 'dataset_sha256': data_hash,
                 'source_identity': fingerprint(provenance()), 'profiles': {}}
        for name, config in candidates(cfg).items():
            builder = GPUProfileBuilder(config, identity, data_hash)
            start = time.monotonic()
            for case in selected:
                print('CALIBRATE', name, case['source_id'], flush=True)
                case_start = time.monotonic()
                blob = engine.prefill(case['context_ids'])
                q = quantize(blob, config)
                builder.add(q, case['source_id'], fingerprint(case['context_ids']))
                torch.cuda.synchronize()
                append(args.output / 'timings.jsonl', {'candidate': name, 'source_id': case['source_id'],
                       'seconds': time.monotonic() - case_start})
                del q, blob
                torch.cuda.empty_cache()
            p = builder.finish()
            path = args.output / (name + '.pt')
            p.save(path)
            index['profiles'][name] = {'identity': p.identity, 'sha256': sha256(path),
                                      'bytes': path.stat().st_size, 'seconds': time.monotonic() - start}
            save(args.output / 'index.json', index)
            del p, builder
            torch.cuda.empty_cache()


def baseline(engine, blob, context, continuation, method, qa=None):
    start = time.monotonic()
    payload, seconds, restore = engine.encode(blob, method)
    decoded = restore(payload)
    score = engine.nll(decoded, context, continuation)
    row = {'method': method, 'actual_bytes': len(payload), 'size_kind': 'actual',
           'nll': score['nll'], 'token_nll': score['token_nll'], 'encode_seconds': seconds}
    if qa is not None:
        row['qa'] = score_answer(engine, decoded, qa)
    row['method_seconds'] = time.monotonic() - start
    del decoded, payload
    return row


def selection(args, cfg):
    chosen = read(args.selection)
    if (chosen['study_identity'] != fingerprint(cfg) or chosen['selection_scope'] != 'validation_only'
            or chosen['source_identity'] != fingerprint(provenance())):
        raise ValueError('Selection does not belong to this frozen validation protocol')
    # Recompute the decision from its immutable validation records, not test data.
    path = args.selection.parent / 'records.jsonl'
    if chosen['validation_records_sha256'] != sha256(path):
        raise ValueError('Validation records changed after selection')
    computed = select_candidates(cfg, records(path), chosen['profile_identities'])
    if any(chosen.get(key) != value for key, value in computed.items()):
        raise ValueError('Frozen validation decision or candidate metadata was changed manually')
    return chosen


def qa_cases(engine, cfg, split='test'):
    result = {}
    suffix = engine.tokenizer.encode('\nUSER: What is the first topic we discussed? Only give me the topic name. Do not summarize yourself. ASSISTANT:', add_special_tokens=False)
    with (ROOT / cfg['dataset']).open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row['test_id'] in cfg['splits'][split]:
                if not row.get('label') or not isinstance(row['label'][0], str):
                    raise ValueError('LongChat first-topic label is required')
                result[row['test_id']] = {'suffix_ids': suffix, 'answers': [row['label'][0]],
                                         'generation_tokens': cfg['qa']['generation_tokens']}
    if set(result) != set(cfg['splits'][split]):
        raise ValueError('Missing QA labels')
    return result


def score_answer(engine, blob, case):
    from experiments.scoring import score_qa
    generated = engine.generate(case['suffix_ids'], blob, case['generation_tokens'])
    return {**generated, **score_qa(generated['prediction'], case['answers'])}


def evaluate(args, cfg, final=False):
    import torch
    from paper_compression.quantization import quantize, reconstruct
    from paper_compression.theory import estimate
    from paper_compression.codec import PaperCodec
    chosen = selection(args, cfg) if final else None
    split = 'test' if final else 'validation'
    names = list(chosen['selected'].values()) if final else list(candidates(cfg))
    with run_folder(args, cfg) as data_hash:
        engine, identity = setup(cfg)
        profiles = load_profiles(args.profiles, cfg, identity, data_hash, names)
        if final and any(profiles[n].identity != chosen['profile_identities'][n] for n in names):
            raise ValueError('Final profile differs from validation profile')
        selected = inputs(engine, cfg, split)
        qa = qa_cases(engine, cfg) if final else {}
        save(args.output / 'model_identity.json', identity)
        save(args.output / 'selection.json', selected)
        if final:
            save(args.output / 'frozen_selection.json', chosen)
            save(args.output / 'qa_selection.json', qa)
        engine.load()
        for case in selected:
            source, context, continuation = case['source_id'], case['context_ids'], case['continuation_ids']
            print('EVALUATE', split, source, len(context), flush=True)
            common = {'split': split, 'source_id': source, 'context_tokens': len(context)}
            blob = engine.prefill(context)
            raw = engine.nll(blob, context, continuation)
            qa_case = qa.get(source) if len(context) == cfg['qa']['context_tokens'] else None
            raw_row = {**common, 'method': 'raw', **raw, 'delta_nll': 0.}
            if qa_case is not None:
                raw_row['qa'] = score_answer(engine, blob, qa_case)
            append(args.output / 'records.jsonl', raw_row)
            uniform_bytes = None
            for method in (('uniform8', 'uniform4', 'cachegen1', 'cachegen2', 'cachegen3') if final else ('uniform8',)):
                row = baseline(engine, blob, context, continuation, method, qa=qa_case)
                if method == 'uniform8':
                    uniform_bytes = row['actual_bytes']
                append(args.output / 'records.jsonl', {**common, **row, 'raw_nll': raw['nll'],
                     'delta_nll': row['nll'] - raw['nll'], 'bytes_relative_uniform8': row['actual_bytes'] / uniform_bytes})
            for name, p in profiles.items():
                method_start = time.monotonic()
                p.require_held_out(source, data_hash)
                q = quantize(blob, p.config)
                size = estimate(q, p, cfg['metadata_budget_bytes'])
                restored = reconstruct(q)
                score = engine.nll(restored, context, continuation)
                row = {**common, 'method': name, 'codec': p.config.to_dict(), **size,
                       'raw_nll': raw['nll'], 'nll': score['nll'], 'token_nll': score['token_nll'],
                       'delta_nll': score['nll'] - raw['nll'], 'profile_identity': p.identity,
                       'profile_bytes': (args.profiles / (name + '.pt')).stat().st_size,
                       'bytes_relative_uniform8': size['estimated_bytes'] / uniform_bytes}
                if qa_case is not None:
                    row['qa'] = score_answer(engine, restored, qa_case)
                if final and source in cfg['real_encoding_source_ids']:
                    codec = PaperCodec(p, identity, args.backend)
                    payload, stats = codec.encode_quantized(q)
                    decoded = codec.decode_quantized(payload, 'cuda')
                    for key in ('anchor', 'delta', 'anchor_scales', 'delta_scales'):
                        a, b = getattr(q, key), getattr(decoded, key)
                        if (a is None) != (b is None) or (a is not None and not torch.equal(a, b)):
                            raise ValueError('Arithmetic roundtrip changed ' + key)
                    actual = reconstruct(decoded)
                    if not torch.equal(actual, restored):
                        raise ValueError('Arithmetic roundtrip changed reconstructed KV')
                    check = engine.nll(actual, context, continuation)
                    if check['token_nll'] != score['token_nll']:
                        raise ValueError('Arithmetic roundtrip changed NLL')
                    error = len(payload) - size['estimated_bytes']
                    row.update(actual_bytes=len(payload), actual_bytes_relative_uniform8=len(payload) / uniform_bytes,
                               actual_minus_estimated_bytes=error, estimate_relative_error=error / len(payload),
                               actual_stream_bytes=stats['arithmetic_stream_bytes'],
                               stream_minus_cross_entropy_bytes=stats['arithmetic_stream_bytes'] - size['cross_entropy_bytes'],
                               quantized_identity=True, kv_identity=True, nll_identity=True,
                               packet_sha256=hashlib.sha256(payload).hexdigest())
                    del payload, decoded, actual, codec
                row['method_seconds'] = time.monotonic() - method_start
                append(args.output / 'records.jsonl', row)
                del q, restored
                torch.cuda.empty_cache()
            del blob
        rows = records(args.output / 'records.jsonl')
        if not final:
            choice = select_candidates(cfg, rows, {n: p.identity for n, p in profiles.items()})
            choice['validation_records_sha256'] = sha256(args.output / 'records.jsonl')
            choice['source_identity'] = fingerprint(provenance())
            save(args.output / 'selected.json', choice)
        summarize(rows, args.output, cfg)


def summarize(rows, output, cfg):
    result = []
    qa_summary = [{'method': method, 'sources': len(group),
                   'mean_em': statistics.mean(r['qa']['em'] for r in group),
                   'mean_f1': statistics.mean(r['qa']['f1'] for r in group)}
                  for method in sorted({r['method'] for r in rows})
                  if (group := [r for r in rows if r['method'] == method and 'qa' in r])]
    for method in sorted({r['method'] for r in rows} - {'raw'}):
        for length in cfg['lengths']:
            group = [r for r in rows if r['method'] == method and r['context_tokens'] == length]
            if not group:
                continue
            reference = sum(r['actual_bytes'] for r in rows if r['method'] == 'uniform8' and r['context_tokens'] == length)
            size = sum(r.get('estimated_bytes', r.get('actual_bytes')) for r in group)
            checked = [r for r in group if 'actual_minus_estimated_bytes' in r]
            profile_bytes = group[0].get('profile_bytes', 0)
            result.append({'method': method, 'context_tokens': length, 'sources': len(group),
                           'mean_delta_nll': statistics.mean(r['delta_nll'] for r in group),
                           'median_delta_nll': statistics.median(r['delta_nll'] for r in group),
                           'bytes_relative_uniform8': size / reference,
                           'size_kind': group[0]['size_kind'], 'profile_bytes': profile_bytes,
                           'bytes_plus_profile_relative_uniform8': (size + profile_bytes) / reference,
                           'actual_checked_sources': len(checked),
                           'mean_estimate_relative_error': statistics.mean(r['estimate_relative_error'] for r in checked) if checked else None,
                           'mean_abs_estimate_relative_error': statistics.mean(abs(r['estimate_relative_error']) for r in checked) if checked else None,
                           'max_abs_estimate_relative_error': max((abs(r['estimate_relative_error']) for r in checked), default=None)})
    save(output / 'summary.json', {'quality_metric': 'held-out continuation NLL',
                                 'qa': qa_summary, 'qa_protocol': cfg['qa'],
                                 'parameter_status': cfg['parameter_status'], 'rows': result})


def plot(args, cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data = read(args.input / 'summary.json')['rows']
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / 'curve_data.json', data)
    for length in cfg['lengths']:
        fig, ax = plt.subplots(figsize=(7, 5))
        for kind, marker in (('core', 'o'), ('no-delta', 's'), ('no-layer', '^')):
            group = [r for r in data if r['context_tokens'] == length and r['method'].startswith(kind + '-')]
            # Original and reconstructed references are separate lines.
            for ref in (('original', 'reconstructed') if kind in ('core', 'no-layer') else ('',)):
                points = sorted([r for r in group if not ref or r['method'].endswith(ref)],
                                key=lambda r: r['bytes_relative_uniform8'])
                ax.plot([r['bytes_relative_uniform8'] for r in points], [r['mean_delta_nll'] for r in points],
                        marker=marker, label=kind + ('/' + ref if ref else ''))
                for r in points:
                    ax.annotate(r['method'].split('-r')[1].split('-')[0].replace('p', '.'),
                                (r['bytes_relative_uniform8'], r['mean_delta_nll']), fontsize=7)
        for r in data:
            if r['context_tokens'] == length and r['method'].startswith(('uniform', 'cachegen')):
                ax.scatter(r['bytes_relative_uniform8'], r['mean_delta_nll'], marker='x', color='black')
                ax.annotate(r['method'], (r['bytes_relative_uniform8'], r['mean_delta_nll']), fontsize=7)
        ax.set(xlabel='Complete bytes / uniform8 bytes (candidates: estimated)',
               ylabel='Mean delta NLL vs raw KV', title=f'{length} context tokens')
        ax.axhline(cfg['selection']['max_mean_delta_nll'], linestyle=':', color='gray', label='validation gate')
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(args.output / f'curve-{length}.png', dpi=200)
        fig.savefig(args.output / f'curve-{length}.pdf')
        plt.close(fig)
    checked = [r for r in records(args.input / 'records.jsonl') if 'actual_minus_estimated_bytes' in r]
    if checked:
        fig, ax = plt.subplots(figsize=(6, 5))
        for name in sorted({r['method'] for r in checked}):
            group = [r for r in checked if r['method'] == name]
            ax.scatter([r['estimated_bytes'] / 1e6 for r in group],
                       [r['actual_bytes'] / 1e6 for r in group], label=name, s=20)
        limit = max(max(r['estimated_bytes'], r['actual_bytes']) for r in checked) / 1e6
        ax.plot([0, limit], [0, limit], '--', color='gray')
        ax.set(xlabel='Estimated complete bytes (MB)', ylabel='Actual complete bytes (MB)')
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(args.output / 'encoding-error.png', dpi=200)
        fig.savefig(args.output / 'encoding-error.pdf')
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'insights', 'calibrate', 'scan', 'final', 'plot', 'ttft', 'adaptive'))
    parser.add_argument('--config', type=Path, default=ROOT / 'paper_compression/configs/core-study.json')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--profiles', type=Path)
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--backend', choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args()
    cfg = validate(read(args.config))
    if args.action == 'plan':
        print(json.dumps({'splits': cfg['splits'], 'candidates': {n: c.to_dict() for n, c in candidates(cfg).items()},
                          'selection': cfg['selection'], 'validated': False}, indent=2))
        return
    if args.output is None:
        parser.error('--output required')
    if args.action in ('scan', 'final', 'ttft', 'adaptive') and args.profiles is None:
        parser.error('--profiles required')
    if args.action in ('final', 'ttft', 'adaptive') and args.selection is None:
        parser.error('--selection required')
    if args.action == 'plot' and args.input is None:
        parser.error('--input required')
    if args.action == 'calibrate':
        calibrate(args, cfg)
    elif args.action in ('scan', 'final'):
        evaluate(args, cfg, final=args.action == 'final')
    elif args.action == 'plot':
        plot(args, cfg)
    elif args.action == 'insights':
        from paper_compression.insights import run
        run(args, cfg)
    else:
        from paper_compression.network import run
        run(args, cfg)


if __name__ == '__main__':
    main()
