"""Joint-v2 data isolation, measured cost collection and experiment lifecycle."""
import dataclasses
import hashlib
import itertools
import json
import math
import os
import platform
import statistics
import time
from pathlib import Path

from .joint_config import REPO_ROOT, repository_path

PARTITIONS = ('test', 'cost', 'residual', 'verification')


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def append_record(path, row, *, durable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        if durable:
            stream.flush()
            os.fsync(stream.fileno())


def source_partition(source_ids, original_test_ids):
    ids = sorted(set(source_ids))
    test = list(original_test_ids)
    if len(set(test)) != 5 or not set(test) <= set(ids):
        raise ValueError('Preserve exactly five available original timing sources')
    rest = [sid for sid in ids if sid not in test]
    if len(rest) < 15:
        raise ValueError('At least 20 independent sources required')
    result = {'test': test, 'cost': rest[:5], 'residual': rest[5:10], 'verification': rest[10:15]}
    validate_partition(result)
    return result


def validate_partition(split):
    seen = set()
    for name in PARTITIONS:
        values = split[name]
        if len(values) != 5 or len(set(values)) != 5 or seen.intersection(values):
            raise ValueError('Source partitions require five distinct, mutually disjoint sources each')
        seen.update(values)


def manifest_for(case, qualities, chunk_tokens):
    total = len(case['context_ids'])
    return {'source_id': case['source_id'], 'total_context_tokens': total,
            'packets_persisted': False, 'source_policy': 'gpu_master',
            'blocks': [{'index': index, 'start': start, 'end': min(start + chunk_tokens, total),
                        'qualities': list(qualities), 'encodings': ['ac_count', 'ac_cdf', 'fixed_bits'],
                        'metadata_candidates': 3 * len(qualities), 'packet_state': 'missing'}
                       for index, start in enumerate(range(0, total, chunk_tokens))]}


def read_sources(base, cfg):
    path = repository_path(base['data']['longchat'])
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if len({r['test_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate LongChat source IDs')
    split = source_partition([r['test_id'] for r in rows], cfg['source_test_ids'])
    return rows, split


def tokenize_cases(rows, source_ids, tokenizer, lengths, continuation_tokens=32):
    cases = []
    for row in rows:
        if row['test_id'] not in source_ids:
            continue
        ids = tokenizer.encode(row['prompt'], add_special_tokens=False)
        for length in lengths:
            if len(ids) < length + continuation_tokens:
                raise ValueError(f'Source {row["test_id"]} too short for T={length}')
            cases.append({'case_id': f'longchat-{row["test_id"]}-{length}', 'source_id': row['test_id'],
                          'context_ids': ids[:length], 'continuation_ids': ids[length:length + continuation_tokens],
                          'suffix_ids': tokenizer.encode('\nQuestion: What is the main topic?\nAnswer:', add_special_tokens=False)})
    return cases


def code_fingerprint(base, cfg, *, device=False):
    """Hash cost-affecting inputs once at stage entry to invalidate stale profiles."""
    paths = list((REPO_ROOT / 'experiments').glob('joint*.py'))
    paths += list((REPO_ROOT / 'LMCache/third_party/cachegen_optim_cuda').glob('*'))
    paths += list((REPO_ROOT / 'joint_cuda').glob('*'))
    paths += [REPO_ROOT / p for p in ('experiments/engine.py', 'experiments/stable.py',
              'mini_repro/transport.py', 'mini_repro/experiment_common.py', 'experiments/scoring.py',
              'LMCache/lmcache/storage_backend/serde/cachegen_encoder.py',
              'LMCache/lmcache/storage_backend/serde/cachegen_decoder.py')]
    digest = hashlib.sha256()
    for path in sorted(p for p in paths if p.is_file() and p.suffix in ('.py', '.cpp', '.cu', '.cuh')):
        digest.update(path.relative_to(REPO_ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    runtime = {'python': platform.python_version(), 'platform': platform.platform(),
               'base': base, 'joint': cfg, 'code_sha256': digest.hexdigest()}
    runtime['data_sha256'] = {name: hashlib.sha256(repository_path(path).read_bytes()).hexdigest()
                              for name,path in base['data'].items()}
    if device:
        import torch
        runtime.update(torch=torch.__version__, cuda=torch.version.cuda,
                       device=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability())
    return hashlib.sha256(json.dumps(runtime, sort_keys=True).encode()).hexdigest(), runtime


def frozen_artifact_digests(run):
    names = ('runtime_profile.json','cost_rows.json','first_rows.json','residual-profile.json',
             'verification.json','source-split.json','joint-config.json','base-config.json')
    return {name:hashlib.sha256((run/name).read_bytes()).hexdigest() for name in names}


def invalidate_freeze(run, manifest):
    manifest['frozen'] = False
    if (run/'frozen.json').exists():
        (run/'frozen.json').unlink()


def verify_frozen_artifacts(run, fingerprint, manifest):
    if not manifest.get('frozen'):
        raise ValueError('Freeze accepted code and profiles before benchmark')
    frozen = json.loads((run/'frozen.json').read_text(encoding='utf-8'))
    if frozen.get('fingerprint') != fingerprint or frozen.get('artifacts') != frozen_artifact_digests(run):
        raise ValueError('frozen_artifact_mismatch')


def quality_conditions(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(tuple(row['condition']),[]).append(row)
    reports = []
    for condition, samples in grouped.items():
        denominator = sum(r['raw_em'] for r in samples)
        source_count = len({r['source_id'] for r in samples})
        retention = sum(r['joint_em'] for r in samples)/denominator if denominator else 0.
        reports.append({'condition':list(condition),'em_retention':retention,
            'baseline_em_positive':denominator,'samples':len(samples),'sources':source_count,
            'passed':denominator > 0 and retention >= .9 and len(samples) >= 20 and source_count >= 5})
    return reports


def summarize(values):
    values = sorted(values)
    if not values or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError('Invalid measured samples')
    return statistics.median(values), values[math.ceil(.9 * len(values)) - 1]


def synchronize():
    import torch
    torch.cuda.synchronize()


def measure(fn):
    synchronize()
    start = time.perf_counter_ns()
    value = fn()
    synchronize()
    return value, (time.perf_counter_ns() - start) / 1e9


def require_gpu():
    import torch
    if torch.__version__ != '2.3.1+cu121' or torch.version.cuda != '12.1':
        raise RuntimeError('Target environment requires torch 2.3.1+cu121 and CUDA 12.1')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA device required for measured stages')
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Expose exactly one GPU with CUDA_VISIBLE_DEVICES')


def eligible_qualities(total):
    return (2, 3) if total == 4096 else (1, 2, 3)


def serializable(value):
    if dataclasses.is_dataclass(value):
        return serializable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items() if k != 'final_kv'}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def stage_guard(manifest, task, *, diagnostic_only=False):
    if manifest.get('synthetic') and task != 'inspect':
        raise ValueError('synthetic profiles cannot drive GPU measurement or formal experiments')
    required = {'fit-residual': ('costs',), 'verify': ('residual',), 'benchmark': ('ready',)}
    if diagnostic_only and task == 'benchmark':
        return
    if task in required and manifest.get('state') not in required[task]:
        raise ValueError(f'{task} requires {required[task][0]} profile')


def calibration_states(cfg):
    """Exhaustive resource-state grid; duplicate fixed-geometry keys are removed."""
    seen = set()
    for total in cfg['context_lengths']:
        for prefix, n, whole in [(p, min(cfg['chunk_tokens'], total-p), False)
                                 for p in range(0, total, cfg['chunk_tokens'])] + [(0, total, True)]:
            for q, encoding, previous, buffer_state, mode in itertools.product(
                    cfg['qualities'], ['whole_text'] if whole else cfg['encodings'] + ['text'],
                    ('none', 'initialize', 'encode', 'restore'), ('first', 'reuse', 'expand'),
                    ('serial', 'pipeline')):
                if encoding in ('text', 'whole_text') and q != 3:
                    continue
                if encoding in ('text','whole_text') and buffer_state != 'first':
                    continue
                tags = (0x12, 0x11) if encoding == 'ac_count' else ((0x10,) if encoding == 'ac_cdf' else (None,))
                for tag in tags:
                    row = {'q': q, 'encoding': encoding, 'N': n, 'prefix': prefix,
                           'previous_gpu': previous, 'buffer_state': 'allocates_each_restore' if encoding in ('text','whole_text') else buffer_state,
                           'mode': mode, 'tag': tag}
                    key = tuple(row.values())
                    if key not in seen:
                        seen.add(key)
                        yield row


def _prime_previous(engine, codec, source, final, state, wire):
    previous = state['previous_gpu']
    if previous == 'initialize':
        import torch
        torch.empty_like(final)
    elif previous == 'encode':
        staged = codec.encode_gpu(source, state['q'], state['encoding'] if state['encoding'] not in ('text','whole_text') else 'fixed_bits',
                                  state['prefix'], final.shape[3])
        codec.seal(staged)
    elif previous == 'restore':
        if state['encoding'] in ('text', 'whole_text'):
            # TEXT states are primed by the real prefix prefill performed by the caller.
            return
        codec.restore(wire, final, state['prefix'])
    synchronize()


def collect_costs(engine, codec, cases, cfg, run, *, state_limit=None):
    """Measure original-KV recipes on cost-only sources, never formal requests."""
    import torch
    from .joint_cache import PacketCache
    from .joint_codec import JointCodec
    from .joint_wire import frame
    primer = JointCodec(engine=engine, device='cuda')
    if 'ac_count' in cfg['encodings']:
        start=time.perf_counter_ns()
        resources=primer.preinitialize_combinadic_table()
        append_record(run/'startup.jsonl',{'stage':'calibration_primer','initialization':'combinadic_read_only',
            'elapsed_ns':time.perf_counter_ns()-start,'resources':resources})
    states = list(calibration_states(cfg))
    if state_limit is not None:
        states = states[:state_limit]
    sources = {}
    # Only five independent sources; reuse prefill within this cost-only stage.
    for case in cases:
        sources[case['case_id']] = measure(lambda c=case: engine.prefill(c['context_ids']))[0]
    result = json.loads((run / 'cost_rows.json').read_text(encoding='utf-8')) if (run / 'cost_rows.json').exists() else []
    done = {tuple(row[k] for k in ('q','encoding','N','prefix','previous_gpu','buffer_state','mode','tag')) for row in result}
    for state_index, state in enumerate(states):
        if tuple(state.values()) in done:
            continue
        eligible = [c for c in cases if len(c['context_ids']) >= state['prefix'] + state['N']]
        chosen = {}
        for case in eligible:
            chosen.setdefault(case['source_id'], case)
        samples = []
        for case in chosen.values():
            raw = sources[case['case_id']]
            total = raw.shape[3]
            start, n = state['prefix'], state['N']
            source = raw[:, :, :, start:start + n, :]
            final = torch.empty_like(raw)
            codec.force_full_b = state['tag'] == 0x11
            primer.force_full_b = codec.force_full_b
            is_text = state['encoding'] in ('text', 'whole_text')

            def reset_stage(stage):
                if state['buffer_state'] == 'reuse':
                    if stage == 'encode':
                        codec.encode(source,state['q'],state['encoding'],start,total)
                    else:
                        codec.restore(prime_wire,final,start)
                if state['buffer_state'] != 'reuse':
                    codec.reset_buffers()
                    if state['buffer_state'] == 'expand' and n > 1:
                        small = source[:, :, :, :max(1, n // 2), :]
                        if stage == 'encode':
                            codec.encode(small, state['q'], state['encoding'], 0, small.shape[3])
                        else:
                            warm_wire = primer.encode(small, state['q'], state['encoding'], 0, small.shape[3])
                            small_final = torch.empty(tuple(small.shape), dtype=torch.float16, device='cuda')
                            codec.restore(warm_wire, small_final, 0)
                actual=codec.workspace_state(state['q'],state['encoding'],n,source.shape[0],source.shape[2],source.shape[4],stage,state['tag'])
                if actual != state['buffer_state']:
                    raise ValueError(f'calibration_workspace_state_unreachable: expected={state["buffer_state"]}, actual={actual}, stage={stage}, state={state}')
                _prime_previous(engine, primer, source, final, state, prime_wire)

            def sample():
                if is_text:
                    previous = state['previous_gpu']
                    if previous == 'initialize':
                        torch.empty_like(final)
                    elif previous == 'encode':
                        prime_source = raw[:, :, :, :min(256,total), :]
                        primer.encode(prime_source,3,'fixed_bits',0,prime_source.shape[3])
                    elif previous == 'restore':
                        engine.prefill(case['context_ids'][:min(256,total)])
                    synchronize()
                    def seal_text():
                        body = engine.text_payload(case['context_ids'][start:start + n])
                        return frame(body, {'token_start': start, 'token_end': start+n, 'method': 'text'})
                    text_wire, seal_s = measure(seal_text)
                    prefix = None if start == 0 else raw[:, :, :, :start, :]
                    from .joint_wire import unframe
                    def restore_text():
                        _, payload = unframe(text_wire)
                        computed = engine.prefill(engine.restore_text(payload), prefix)
                        final[:, :, :, start:start+n, :] = computed[:, :, :, start:start+n, :]
                    _, restore_s = measure(restore_text)
                    return {'g': 0., 'c': seal_s, 'r': restore_s, 'size': len(text_wire), 'access': 0.}
                reset_stage('encode')
                fields, gpu_s = measure(lambda: codec.encode_gpu(source, state['q'], state['encoding'], start, total))
                begin = time.perf_counter_ns()
                wire = codec.seal(fields)
                index = PacketCache()
                index.put('packet', wire)
                seal_s = (time.perf_counter_ns() - begin) / 1e9
                reset_stage('restore')
                _, restore_s = measure(lambda: codec.restore(wire, final, start))
                # Real immutable packet access; no body copy.
                begin = time.perf_counter_ns()
                with index.acquire('packet'):
                    pass
                access_s = (time.perf_counter_ns() - begin) / 1e9
                return {'g': gpu_s, 'c': seal_s, 'r': restore_s, 'size': len(wire), 'access': access_s}

            # Prime real task transitions before each measured service, outside its service interval.
            if not is_text:
                prime_wire = primer.encode(source, state['q'], state['encoding'], start, total)
            else:
                prime_wire = b''
            warmup = cfg['calibration_warmup'] if not samples else 0
            for repeat in range(warmup + cfg['calibration_repeats_per_source']):
                row = sample()
                if repeat >= warmup:
                    row.update(source_id=case['source_id'], repeat=repeat - warmup)
                    samples.append(row)
            del final, source
        if len(samples) != 20 or len({r['source_id'] for r in samples}) != 5:
            raise ValueError('Each cost state requires exactly five sources and twenty observations')
        summary = dict(state)
        for short in ('g', 'c', 'r', 'access'):
            summary[f'{short}50_s'], summary[f'{short}90_s'] = summarize([r[short] for r in samples])
        summary['size50'], summary['size90'] = summarize([r['size'] for r in samples])
        summary.update(samples=20, source_ids=sorted({r['source_id'] for r in samples}))
        result.append(summary)
        append_record(run / 'cost-observations.jsonl', {'state': state, 'samples': samples})
        save_json(run / 'cost_rows.json', result)
    codec.force_full_b = False
    del sources
    return result


def collect_first_costs(engine, cases, cfg, run):
    rows = []
    from mini_repro.experiment_common import OBJECTS
    suffixes = {len(cases[0]['suffix_ids']): cases[0]['suffix_ids']}
    for item in OBJECTS:
        ids = engine.tokenizer.encode(f'\nQuestion: Who donated the {item}?\nAnswer:', add_special_tokens=False)
        suffixes.setdefault(len(ids), ids)
    for total in cfg['context_lengths']:
        selected = [c for c in cases if len(c['context_ids']) == total]
        for suffix_count, suffix in suffixes.items():
            values = []
            for case in selected:
                raw = measure(lambda: engine.prefill(case['context_ids']))[0]
                warmup = cfg['calibration_warmup'] if not values else 0
                for repeat in range(warmup + cfg['calibration_repeats_per_source']):
                    _, elapsed = measure(lambda: engine.first(case['context_ids'], suffix, raw))
                    if repeat >= warmup:
                        values.append({'source_id': case['source_id'], 'seconds': elapsed})
                del raw
            h50, h90 = summarize([r['seconds'] for r in values])
            rows.append({'T': total, 'suffix_tokens': suffix_count,
                         'h50_s': h50, 'h90_s': h90, 'samples': values})
    save_json(run / 'first_rows.json', rows)
    return rows


def bandwidth_trace(name, bandwidth_gbps):
    """Limiter-only wall-clock trajectory; never exposed to the selector."""
    if name == 'stable':
        return bandwidth_gbps * 1e9
    if name == 'step':
        return lambda elapsed_s: (bandwidth_gbps if elapsed_s < .05 else bandwidth_gbps / 10) * 1e9
    raise ValueError('Unknown bandwidth trajectory')
