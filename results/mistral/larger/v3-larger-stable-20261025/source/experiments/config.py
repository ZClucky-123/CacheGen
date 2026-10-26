"""Model-independent, versioned experiment configuration; no ML imports."""
import copy
import json
import math
import re
from pathlib import Path

PROTOCOL = 'cachegen-shared-v3'
METHODS = ('raw', 'uniform4', 'uniform8', 'cachegen1', 'cachegen2', 'cachegen3')
STRATEGIES = ('adaptive', 'adaptive_history', 'adaptive_static', 'fixed1', 'fixed2', 'fixed3', 'text')
TRANSPORT_MODES = ('whole', 'chunk_serial', 'chunk_pipeline')
TASK_KINDS = ('quality', 'ttft', 'adaptive', 'ablation')
DEFAULT_EXECUTION = {'linear_tile_tokens': 0, 'explicit_causal_mask': False,
                     'sdpa_backend': 'auto', 'fp16_reduced_precision_reduction': True}
MODEL_SPECS = {
    'tinyllama': {
        'id': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        'revision': 'fe8a4ea1ffedaf415f4da2f062534de366a451e6',
        'weight_loading': 'fp16', 'layers': 22, 'kv_heads': 4, 'head_dim': 64,
        'context_limit': 2048, 'key_boundaries': [7, 14, 22], 'value_boundary': 2,
    },
    'mistral': {
        'id': 'mistral-community/Mistral-7B-v0.2',
        'revision': '80cf49b18de6354fd8a6d1e81c177dd830e05eea',
        'weight_loading': 'fastchat8', 'layers': 32, 'kv_heads': 8, 'head_dim': 128,
        'context_limit': 32768, 'key_boundaries': [10, 20, 32], 'value_boundary': 2,
    },
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_config(path, model=None):
    path = Path(path)
    cfg = json.loads(path.read_text(encoding='utf-8'))
    if 'extends' in cfg:
        base = load_config(path.parent / cfg.pop('extends'))
        cfg = base | cfg
    return validate_config(cfg, model)


def validate_config(cfg, model=None):
    """Apply the same checks to files, adapter overrides and migrated profiles."""
    cfg = copy.deepcopy(cfg)
    execution = cfg.setdefault('execution', copy.deepcopy(DEFAULT_EXECUTION))
    require(set(execution) == set(DEFAULT_EXECUTION), 'Provide all execution settings')
    require(type(execution['linear_tile_tokens']) is int and execution['linear_tile_tokens'] in (0, 32, 64, 128, 256, 512),
            'Invalid stable linear tile')
    require(type(execution['explicit_causal_mask']) is bool and type(execution['fp16_reduced_precision_reduction']) is bool,
            'Execution switches must be boolean')
    require(execution['sdpa_backend'] in ('auto', 'flash', 'efficient', 'math'), 'Invalid SDPA backend')
    if execution['linear_tile_tokens']:
        require(execution['explicit_causal_mask'] and execution['sdpa_backend'] == 'efficient'
                and not execution['fp16_reduced_precision_reduction'], 'Stable tiling requires explicit masks, efficient SDPA and FP32 reduction')
    require(cfg.get('protocol') == PROTOCOL, f'Expected protocol {PROTOCOL}; legacy profiles require migration')
    if model:
        cfg['model'] = model
    spec = cfg['model']
    if isinstance(spec, str):
        require(spec in MODEL_SPECS, f'Unknown model: {spec}')
        cfg['model'] = copy.deepcopy(MODEL_SPECS[spec]) | {'name': spec}
    spec = cfg['model']
    require(isinstance(spec, dict), 'model must be a registered name or complete adapter mapping')
    require(cfg.get('scale') in ('minimal', 'small', 'larger'), 'Invalid result scale')
    for key in ('name', 'id', 'revision', 'weight_loading', 'layers', 'kv_heads', 'head_dim',
                'context_limit', 'key_boundaries', 'value_boundary'):
        require(key in spec, f'Missing model.{key}')
    require(spec['weight_loading'] in ('fp16', 'fastchat8'), 'Unsupported weight loading')
    require(len(spec['revision']) == 40 and all(c in '0123456789abcdef' for c in spec['revision']), 'Pin a full model revision')
    for key in ('layers', 'kv_heads', 'head_dim', 'context_limit'):
        require(type(spec[key]) is int and spec[key] > 0, f'Invalid model.{key}')
    boundaries = spec['key_boundaries']
    require(re.fullmatch(r'[a-z][a-z0-9_-]*', spec['name']) is not None, 'Invalid adapter result name')
    require(len(boundaries) == 3 and all(type(v) is int for v in boundaries) and boundaries == sorted(boundaries) and boundaries[0] > 0
            and boundaries[-1] == spec['layers'], 'Codec boundaries must cover model layers')
    require(0 <= spec['value_boundary'] <= spec['layers'], 'Invalid value boundary')
    require(type(cfg['seed']) is int and cfg['seed'] >= 0, 'Invalid seed')
    require(set(cfg['quality']['methods']) == set(METHODS) and len(cfg['quality']['methods']) == 6,
            'Quality requires all six methods once')
    require(set(cfg['ttft']['methods']).issubset({'text', *METHODS})
            and {'text', 'uniform8', 'cachegen2', 'cachegen3'}.issubset(cfg['ttft']['methods'])
            and len(set(cfg['ttft']['methods'])) == len(cfg['ttft']['methods']), 'Invalid TTFT methods')
    require(tuple(cfg['adaptive']['strategies']) == STRATEGIES, 'Preserve original controls and the history variant')
    require(cfg['adaptive']['mode'] in ('socket_serial', 'trace_estimate'), 'Invalid adaptive timing mode')
    require(cfg['adaptive']['mode'] == 'socket_serial', 'Protocol v3 history feedback requires measured serial transfers')
    require(cfg['adaptive']['throughput_estimator'] == 'previous_completed_message', 'Unsupported throughput estimator')
    initial = cfg['adaptive']['initial_bandwidth_gbps']
    require(type(initial) in (int, float) and math.isfinite(initial) and initial > 0, 'Fix positive initial bandwidth before measurement')
    modes = cfg['ttft']['modes']
    require(isinstance(modes, list) and set(modes) == set(TRANSPORT_MODES) and len(modes) == 3,
            'Keep whole reference and both chunk controls')
    sizes = cfg['ttft']['chunk_sizes']
    require(isinstance(sizes, list) and sizes and len(set(sizes)) == len(sizes)
            and all(type(v) is int and v > 0 for v in sizes), 'Invalid application chunk sizes')
    if execution['linear_tile_tokens']:
        require(all(size % execution['linear_tile_tokens'] == 0 for size in sizes + [cfg['adaptive']['chunk_size']]),
                'Application chunks must align with the stable linear tile')
    capacity = cfg['transport']['queue_capacity']
    require(type(capacity) is int and capacity > 0, 'Invalid bounded receive queue capacity')
    quantum = cfg['transport']['pacing_quantum_bytes']
    require(quantum is None or type(quantum) is int and quantum > 0, 'Invalid pacing quantum')
    ablation = cfg['ablation']
    require(type(ablation['enabled']) is bool, 'ablation.enabled must be boolean')
    for field in ('samples', 'qa_per_position', 'warmup', 'repeats'):
        require(type(ablation[field]) is int and ablation[field] > 0, f'Invalid ablation.{field}')
    require(ablation['samples'] <= cfg['quality']['samples'] and ablation['qa_per_position'] <= cfg['quality']['qa_per_position'],
            'Ablation uses a declared subset of quality cases')
    require(isinstance(ablation['lengths'], list) and ablation['lengths'] and len(set(ablation['lengths'])) == len(ablation['lengths'])
            and set(ablation['lengths']).issubset(cfg['quality']['lengths']), 'Invalid ablation lengths')
    require(isinstance(ablation['precisions'], list) and len(set(ablation['precisions'])) == len(ablation['precisions'])
            and {'uniform16', 'uniform32', 'layered2'}.issubset(ablation['precisions'])
            and set(ablation['precisions']).issubset({'uniform16', 'uniform32', 'layered1', 'layered2'}),
            'Declare both uniform bounds and at least layered2')
    require(ablation['entropy'] == ['packed', 'arithmetic'], 'Keep paired entropy controls')
    for section, keys in {
        'quality': ('samples', 'continuation', 'qa_per_position', 'max_new_tokens'),
        'triviaqa': ('samples', 'max_prompt_tokens', 'max_new_tokens'),
        'ttft': ('samples', 'repeats'),
        'adaptive': ('cases', 'length', 'chunk_size', 'traces', 'repeats', 'max_new_tokens'),
        'calibration': ('repeats',),
    }.items():
        for key in keys:
            require(type(cfg[section][key]) is int and cfg[section][key] > 0, f'{section}.{key} must be a positive integer')
    require(cfg['quality']['continuation'] >= 2, 'Need at least two continuation tokens')
    require(cfg['quality']['qa_per_position'] <= 10 and cfg['adaptive']['cases'] <= 10,
            'Only ten independent controlled facts exist; do not count duplicates as new facts')
    for section in ('ttft', 'adaptive'):
        require(type(cfg[section]['warmup']) is int and cfg[section]['warmup'] >= 1, 'At least one warmup is required')
    for section, keys in {'quality': ('lengths',), 'ttft': ('lengths', 'bandwidth_gbps'),
                          'adaptive': ('bandwidth_scales', 'slos'), 'calibration': ('payload_bytes',)}.items():
        for key in keys:
            values = cfg[section][key]
            require(isinstance(values, list) and values and len(set(values)) == len(values)
                    and all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in values),
                    f'Invalid {section}.{key}')
            if key in ('lengths', 'payload_bytes'):
                require(all(type(v) is int for v in values), f'{section}.{key} requires integers')
    for key in ('longchat', 'triviaqa', 'bandwidth'):
        require(isinstance(cfg['data'][key], str), f'Missing data.{key}')
    require(cfg['quality_tolerances'] == {'max_nll_delta': 0.02, 'minimum_retention': 0.9},
            'Protocol v3 preserves the existing quality tolerances')
    require(cfg['scoring'] == {'answer': 'author_first_lf_line', 'normalization': 'standard_em_f1',
                               'aliases': 'dataset_only', 'stop': 'eos_or_token_limit'}, 'Scoring is frozen by protocol')
    return cfg


def jobs(cfg):
    return [{'name': 'quality', 'kind': 'quality'}, {'name': 'ttft', 'kind': 'ttft'}] + [
        {'name': f'adaptive-scale-{scale}-slo-{slo}', 'kind': 'adaptive', 'scale': scale, 'slo': slo}
        for scale in cfg['adaptive']['bandwidth_scales'] for slo in cfg['adaptive']['slos']] + (
            [{'name': 'ablation', 'kind': 'ablation'}] if cfg['ablation']['enabled'] else [])


def ttft_variants(cfg, context_tokens):
    for mode in cfg['ttft']['modes']:
        for size in ([context_tokens] if mode == 'whole' else cfg['ttft']['chunk_sizes']):
            yield mode, size


def ablation_cases(cfg, selected):
    p = cfg['ablation']
    natural_sources = list(dict.fromkeys(c['source_id'] for c in selected['quality']))[:p['samples']]
    for kind in ('quality', 'controlled'):
        for case in selected[kind]:
            if case['context_tokens'] in p['lengths'] and (
                    case['source_id'] in natural_sources if kind == 'quality' else case['source_id'] < p['qa_per_position']):
                yield kind, case
