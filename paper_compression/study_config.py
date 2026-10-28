"""Pre-registered split, candidate construction, and validation-only selection."""
from dataclasses import replace
import math
from .config import CodecConfig, fingerprint


def validate(cfg):
    if cfg.get('protocol') != 'paper-core-study-v2':
        raise ValueError('Expected paper-core-study-v2')
    required = {'calibration': list(range(30, 50)), 'validation': list(range(20, 30)),
                'test': list(range(20))}
    if cfg['splits'] != required:
        raise ValueError('Study fixes calibration 30..49, validation 20..29, test 0..19')
    if cfg['model']['weight_loading'] != 'fp16':
        raise ValueError('Paired study requires FP16 model weights')
    if not cfg['vector_ranges'] or any(not math.isfinite(v) or v <= 0 for v in cfg['vector_ranges']):
        raise ValueError('Invalid vector ranges')
    if len(set(cfg['vector_ranges'])) != len(cfg['vector_ranges']):
        raise ValueError('Duplicate vector range')
    if not cfg['lengths'] or any(type(n) is not int or n < 2 or
            n + cfg['continuation_tokens'] > cfg['model']['context_limit'] for n in cfg['lengths']):
        raise ValueError('Invalid context lengths')
    if len(set(cfg['lengths'])) != len(cfg['lengths']):
        raise ValueError('Duplicate context lengths')
    if (not math.isfinite(cfg['selection']['max_mean_delta_nll'])
            or cfg['selection']['max_mean_delta_nll'] < 0 or type(cfg['metadata_budget_bytes']) is not int
            or cfg['metadata_budget_bytes'] < 0):
        raise ValueError('Invalid selection gate or byte budget')
    if (type(cfg['calibration_tokens']) is not int or not 2 <= cfg['calibration_tokens'] <= cfg['model']['context_limit']
            or type(cfg['continuation_tokens']) is not int or cfg['continuation_tokens'] < 2
            or cfg['qa']['context_tokens'] not in cfg['lengths'] or cfg['qa']['generation_tokens'] < 1):
        raise ValueError('Invalid calibration, continuation or QA length')
    if not set(cfg['real_encoding_source_ids']) <= set(required['test']):
        raise ValueError('Real coding checks must use predeclared test IDs')
    p = cfg['network']
    if (not p['source_ids'] or not set(p['source_ids']) <= set(required['test']) or p['tokens'] not in cfg['lengths']
            or type(p['chunk_tokens']) is not int or p['chunk_tokens'] < 1
            or type(p['repeats']) is not int or p['repeats'] < 1
            or type(p['probe_bytes']) is not int or p['probe_bytes'] < 1
            or not 0 <= p['safety_fraction'] < 1 or p['min_reserve_seconds'] < 0
            or p['slo_seconds'] != [0.5, 1.0]):
        raise ValueError('Invalid network protocol')
    if cfg['insights']['tokens'] != cfg['calibration_tokens']:
        raise ValueError('Token-position grouping requires matched calibration/validation lengths')
    nchunks = (p['tokens'] + p['chunk_tokens'] - 1) // p['chunk_tokens']
    if (not p['bandwidth_gbps'] or any(not math.isfinite(v) or v <= 0 for v in p['bandwidth_gbps'])
            or not p['traces_gbps'] or any(len(trace) != nchunks or any(not math.isfinite(v) or v <= 0
               for v in trace) for trace in p['traces_gbps'])):
        raise ValueError('Invalid bandwidth or trace length')
    return cfg


def candidates(cfg):
    base = CodecConfig(layers=cfg['model']['layers'], vector_range=cfg['vector_ranges'][0])
    result = {}
    for value in cfg['vector_ranges']:
        tag = str(value).replace('.', 'p')
        for ref in ('original', 'reconstructed'):
            result[f'core-r{tag}-{ref}'] = replace(base, vector_range=value, anchor_reference=ref)
            result[f'no-layer-r{tag}-{ref}'] = replace(base, vector_range=value, anchor_reference=ref,
                                                    delta_bin_widths=(1., 1., 1.))
        result[f'no-delta-r{tag}'] = replace(base, vector_range=value, use_delta=False)
    return result


def family(name):
    return 'core' if name.startswith('core-') else 'no-delta' if name.startswith('no-delta-') else 'no-layer'


def select_candidates(cfg, rows, profiles):
    """Select smallest aggregate estimate meeting the gate at EVERY length."""
    expected = {(source, length) for source in cfg['splits']['validation'] for length in cfg['lengths']}
    summaries = []
    for name in candidates(cfg):
        group = [r for r in rows if r['method'] == name]
        keys = [(r['source_id'], r['context_tokens']) for r in group]
        if len(keys) != len(expected) or set(keys) != expected:
            raise ValueError('Incomplete/duplicated validation matrix: ' + name)
        if any(not math.isfinite(r['delta_nll']) or not math.isfinite(r['estimated_bytes'])
               or r['estimated_bytes'] <= 0 for r in group):
            raise ValueError('Nonfinite quality/size or invalid byte estimate')
        means = {str(n): sum(r['delta_nll'] for r in group if r['context_tokens'] == n) /
                 len(cfg['splits']['validation']) for n in cfg['lengths']}
        feasible = all(v <= cfg['selection']['max_mean_delta_nll'] for v in means.values())
        summaries.append({'method': name, 'family': family(name), 'mean_delta_nll_by_length': means,
                          'estimated_bytes': sum(r['estimated_bytes'] for r in group), 'feasible': feasible})
    chosen = {}
    configs = candidates(cfg)
    for kind in ('core', 'no-delta', 'no-layer'):
        options = [r for r in summaries if r['family'] == kind and r['feasible']]
        if kind == 'no-layer':
            options = [r for r in options if configs[r['method']].anchor_reference ==
                       configs[chosen['core']].anchor_reference]
        if not options:
            raise ValueError('No validation candidate meets predeclared gate: ' + kind)
        chosen[kind] = min(options, key=lambda r: (r['estimated_bytes'], r['method']))['method']
    return {'study_identity': fingerprint(cfg), 'selected': chosen, 'candidates': summaries,
            'profile_identities': {name: profiles[name] for name in chosen.values()},
            'selection_scope': 'validation_only', 'gate': cfg['selection'],
            'parameter_status': 'interpretation_pending_verification'}
