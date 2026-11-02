"""Joint-v2 calibration, independent acceptance and paired experiments."""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

from .joint_config import load_joint_config, repository_path, run_directory, REPO_ROOT
from .joint_records import benchmark_directory, load_benchmark_checkpoint, record_benchmark_completion
from .joint_calibration import (read_sources, save_json, append_record, code_fingerprint,
    stage_guard, calibration_states, tokenize_cases, collect_costs, collect_first_costs,
    require_gpu, manifest_for, eligible_qualities, serializable, measure, summarize,
    bandwidth_trace, frozen_artifact_digests, invalidate_freeze, verify_frozen_artifacts, quality_conditions)


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-config', default='experiments/configs/larger-mistral.json')
    p.add_argument('--config', default='optimization/configs/joint-v2.json')
    p.add_argument('--task', choices=('calibrate', 'check', 'benchmark', 'inspect'), required=True)
    p.add_argument('--phase', choices=('fit-residual', 'verify'))
    p.add_argument('--run-dir')
    p.add_argument('--run-id')
    p.add_argument('--plan', action='store_true', help='print scope without creating files or loading ML')
    p.add_argument('--freeze', action='store_true', help='freeze an independently accepted runtime profile')
    p.add_argument('--state-limit', type=int, help='small diagnostic calibration only; cannot become ready')
    p.add_argument('--fixed-quality', type=int, choices=(1, 2, 3))
    p.add_argument('--diagnostic-only', action='store_true')
    p.add_argument('--group', action='append', help='restrict benchmark groups; repeatable')
    args = p.parse_args(argv)
    if args.task == 'check' and not args.phase:
        p.error('check requires --phase fit-residual|verify')
    if args.freeze and args.task != 'inspect':
        p.error('--freeze is an inspect action after verification')
    if args.diagnostic_only and args.fixed_quality is None:
        p.error('--diagnostic-only requires --fixed-quality')
    if args.state_limit is not None and args.state_limit <= 0:
        p.error('--state-limit must be positive')
    return args


def groups(qs):
    for q in qs:
        yield {'name': f'original-q{q}', 'original': f'cachegen{q}', 'fixed_quality': q}
        for encoding in ('ac_count', 'ac_cdf', 'fixed_bits'):
            yield {'name': f'{encoding}-q{q}', 'encodings': [encoding], 'fixed_quality': q}
        yield {'name': f'joint-two-q{q}', 'encodings': ['ac_count', 'fixed_bits'], 'fixed_quality': q}
        yield {'name': f'joint-three-q{q}', 'fixed_quality': q}
    yield {'name': 'joint-adaptive'}
    yield {'name': 'whole-text', 'route': 'whole_text'}
    yield {'name': 'uniform4-whole', 'original': 'uniform4', 'whole': True}
    yield {'name': 'original-adaptive-history', 'legacy_adaptive': True}


def plan_summary(cfg, split, run, args):
    state_count = len(list(calibration_states(cfg)))
    count = 0
    for total in cfg['context_lengths']:
        selected = list(groups([args.fixed_quality] if args.fixed_quality else eligible_qualities(total)))
        if args.group:
            selected = [g for g in selected if g['name'] in args.group]
        count += sum(1 for g in selected for m in ('serial','pipeline') for c in cfg.get('benchmark_cache_conditions',cfg['cache_conditions'])
                     if benchmark_condition_supported(g,m,c)) * 5 * cfg['benchmark_repeats_per_source'] * len(cfg['bandwidth_gbps']) * len(cfg['benchmark_slos_seconds'])
    step_count=0
    if 4096 in cfg['context_lengths'] and args.fixed_quality in (None,2):
        paired=[g for g in groups([2]) if not g.get('original') and not g.get('legacy_adaptive') and g.get('fixed_quality') == 2]
        if args.group:
            paired=[g for g in paired if g['name'] in args.group]
        step_count=sum(1 for g in paired for m in ('serial','pipeline') for c in cfg.get('benchmark_cache_conditions',cfg['cache_conditions'])
                       if benchmark_condition_supported(g,m,c))*5*cfg['benchmark_repeats_per_source']*len(cfg['bandwidth_gbps'])*len(cfg['benchmark_slos_seconds'])
    return {'run_dir': str(run), 'source_split': split, 'calibration_states': state_count,
            'calibration_measured_samples': state_count * 20,
            'calibration_warmup_samples': state_count * cfg['calibration_warmup'],
            'formal_requests_stable': count, 'formal_requests_step':step_count,
            'additional_bandwidth_step': 'q2/4K paired observations',
            'execution': 'not_started', 'gpu_validation': 'pending_target_device'}


def benchmark_condition_supported(group, mode, condition):
    if group.get('legacy_adaptive') and mode != 'serial':
        return False
    if group.get('whole') and mode != 'serial':
        return False
    if condition == 'mixed' and (group.get('whole') or group.get('legacy_adaptive') or group.get('route') == 'whole_text'):
        return False
    if condition in ('quality_change','representation_miss'):
        if group.get('fixed_quality') is None or group.get('whole') or group.get('legacy_adaptive') or group.get('route') == 'whole_text':
            return False
        if condition == 'representation_miss':
            if group.get('original') or len(group.get('encodings',('ac_count','ac_cdf','fixed_bits'))) == 3:
                return False
    return True


def seed_request_options(group, condition, total):
    """A single real prior route; no candidate/package prepopulation."""
    options = {key:group[key] for key in ('encodings','fixed_quality','route') if key in group}
    if 'fixed_quality' in options:
        options['quality_mode'] = 'fixed'
    if condition in ('cold','warm','mixed'):
        return None if condition == 'cold' else options
    if not benchmark_condition_supported(group,'serial',condition):
        raise ValueError('Cache condition is inapplicable to this target group')
    q = group['fixed_quality']
    methods = tuple(group.get('encodings',('ac_count','ac_cdf','fixed_bits')))
    if condition == 'quality_change':
        others = [value for value in eligible_qualities(total) if value != q]
        seed_q = min(others,key=lambda value:(abs(value-q),value))
        if group.get('original'):
            return {'original':f'cachegen{seed_q}','fixed_quality':seed_q}
        return {'quality_mode':'fixed','fixed_quality':seed_q,'encodings':[methods[0]]}
    seed_m = next(m for m in ('ac_count','ac_cdf','fixed_bits') if m not in methods)
    return {'quality_mode':'fixed','fixed_quality':q,'encodings':[seed_m]}


def seed_runtime(runtime,ctx,case,cfg,slo,bw,group,condition,args):
    options = seed_request_options(group,condition,len(case['context_ids']))
    if options is None:
        return None
    seeded = runtime.run(ctx,case['suffix_ids'],slo,bandwidth_bps=bw,
        diagnostic_only=args.diagnostic_only,validation_only=args.diagnostic_only,**options)
    if condition == 'mixed':
        _prune_to_mixed(runtime.caches[ctx.context_id])
    return serializable(seeded)


def initialize_codec(codec,encodings,run,label):
    if 'ac_count' not in encodings:
        return
    start = time.perf_counter_ns()
    resources = codec.preinitialize_combinadic_table()
    append_record(run/'startup.jsonl',{'stage':label,'initialization':'combinadic_read_only',
        'elapsed_ns':time.perf_counter_ns()-start,'resources':resources})


def validate_joint_model(base):
    from .config import MODEL_SPECS
    spec, expected = base['model'], MODEL_SPECS['mistral']
    if any(spec.get(key) != expected[key] for key in
           ('id','revision','weight_loading','layers','kv_heads','head_dim')):
        raise ValueError('Joint-v2 requires the pinned Mistral FastChat8 model and (32,8,128) KV geometry')


def startup_encodings(cfg,args):
    if args.task != 'benchmark' or not args.group:
        return cfg['encodings']
    enabled=set()
    for group in groups((1,2,3)):
        if group['name'] not in args.group or group.get('original') or group.get('legacy_adaptive') or group.get('route') == 'whole_text':
            continue
        enabled.update(group.get('encodings',cfg['encodings']))
        for condition in cfg.get('benchmark_cache_conditions',cfg['cache_conditions']):
            if benchmark_condition_supported(group,'serial',condition):
                seed=seed_request_options(group,condition,4096 if group.get('fixed_quality') in (2,3) else 1024)
                if seed:
                    enabled.update(seed.get('encodings',()))
    return enabled


def profile_for(run, fingerprint, *, ready=False):
    from .joint_profiles import load_runtime_profile
    return load_runtime_profile(run, fingerprint, require_ready=ready)


def _record_decisions(row, mode):
    result = []
    for decision in row['decisions']:
        execution_mode = mode
        route = decision.get('route', row.get('route', 'chunks'))
        total = row['total_context_tokens']
        cache_class = decision.get('cache_class', row['cache_condition'])
        result.append({**decision, 'request_id': row['request_id'], 'source_id': row['source_id'],
            'execution_mode': execution_mode, 'route': route, 'total_context_tokens': total,
            'cache_class': cache_class, 'first_token_ns': row['first_token_ns'],
            'deadline_ns': row['deadline_ns'], 'success': True,
            'nll_delta': row['nll_delta'], 'em_retention': row.get('em_retention'),
            'bandwidth_class': row.get('trajectory', 'stable'),
            'quality_condition': [mode,row.get('route','chunks'),total,row['cache_condition'],row.get('trajectory','stable')]})
    return result


def _prune_to_mixed(cache):
    # Retain actual recorded packet entries only; never synthesize missing routes.
    with cache.lock:
        for index, key in enumerate(list(cache.entries)):
            if index % 2 == 0 and not cache.entries[key].pins:
                cache.resident_bytes -= len(cache.entries.pop(key).wire)
                cache.evictions += 1
    cache.on_change()


def check_requests(engine, codec, profile, cfg, cases, run, phase):
    from .joint_runtime import Runtime
    records, requests = [], []
    for mode in ('serial', 'pipeline'):
        local = dict(cfg, execution_mode=mode, generation_lookahead=0 if mode == 'serial' else 1)
        for case in cases:
            raw, source_s = measure(lambda: engine.prefill(case['context_ids']))
            baseline_nll = engine.nll(raw, case['context_ids'], case['continuation_ids'])['nll']
            for route in ('chunks', 'whole_text'):
                for cache_condition in cfg['cache_conditions']:
                    if route == 'whole_text' and cache_condition == 'mixed':
                        continue
                    for trace in cfg['bandwidth_trajectories']:
                        for repeat in range(cfg['check_repeats_per_source']):
                            runtime = Runtime(engine, codec, profile, local)
                            ctx = runtime.register_context(case['context_ids'], raw, case['case_id'])
                            kwargs = {'route': route}
                            try:
                                if cache_condition != 'cold':
                                    # A real prior request is the only producer of warm packets.
                                    runtime.run(ctx, case['suffix_ids'], cfg['check_slo_s'],
                                        validation_only=True, residual_mode='collect' if phase == 'fit-residual' else 'normal', **kwargs)
                                    if cache_condition == 'mixed':
                                        _prune_to_mixed(runtime.caches[ctx.context_id])
                                result = runtime.run(ctx, case['suffix_ids'], cfg['check_slo_s'],
                                    bandwidth_bps=bandwidth_trace(trace, 1.), validation_only=True,
                                    residual_mode='collect' if phase == 'fit-residual' else 'normal', **kwargs)
                                row = serializable(result)
                                row.update(source_id=case['source_id'], case_id=case['case_id'],
                                    total_context_tokens=len(case['context_ids']), cache_condition=cache_condition,
                                    route=route, execution_mode=mode, trajectory=trace, repeat=repeat,
                                    validation_only=True, source_prefill_s=source_s, success=True,
                                    nll_delta=engine.nll(result['final_kv'], case['context_ids'], case['continuation_ids'])['nll'] - baseline_nll)
                                requests.append(row)
                                records.extend(_record_decisions(row, mode))
                                append_record(run / phase / 'requests.jsonl', row)
                            except Exception as exc:
                                append_record(run / phase / 'failures.jsonl', {'source_id': case['source_id'],
                                    'case_id': case['case_id'], 'error': type(exc).__name__, 'message': str(exc),
                                    'joint_failure':serializable(getattr(exc,'joint_failure',None))})
                                raise
                            finally:
                                runtime.close_context(ctx)
            del raw
    return requests, records


def qa_acceptance(engine, codec, profile, cfg, source_ids, run):
    """Original framework controlled facts and exact-match scorer on real generations."""
    from .joint_runtime import Runtime
    from .scoring import score_qa
    from mini_repro.experiment_common import build_positioned_case
    rows = []
    for total in cfg['context_lengths']:
        for source_id in source_ids:
            for repeat in range(cfg['check_repeats_per_source']):
                position = ('start','middle','end')[repeat % 3]
                case = build_positioned_case(engine.tokenizer, int(source_id), total, position)
                ids = case['input_ids']
                suffix = engine.tokenizer.encode(f'\nQuestion: {case["question"]}\nAnswer:', add_special_tokens=False)
                raw = measure(lambda: engine.prefill(ids))[0]
                reference = engine.generate(suffix, raw, cfg['max_new_tokens'])
                reference_em = score_qa(reference['prediction'], [case['answer']])['em']
                for mode in ('serial','pipeline'):
                    local = dict(cfg,execution_mode=mode)
                    for route in ('chunks','whole_text'):
                        for condition in cfg['cache_conditions']:
                            if route == 'whole_text' and condition == 'mixed':
                                continue
                            for trace in cfg['bandwidth_trajectories']:
                                runtime = Runtime(engine, codec, profile, local)
                                ctx = runtime.register_context(ids, raw, f'qa-{source_id}-{total}-{position}')
                                try:
                                    if condition != 'cold':
                                        runtime.run(ctx,suffix,cfg['check_slo_s'],route=route,validation_only=True)
                                        if condition == 'mixed':
                                            _prune_to_mixed(runtime.caches[ctx.context_id])
                                    result = runtime.run(ctx,suffix,cfg['check_slo_s'],route=route,
                                        bandwidth_bps=bandwidth_trace(trace,1.),validation_only=True)
                                    generated = engine.generate(suffix,result['final_kv'],cfg['max_new_tokens'])
                                    em = score_qa(generated['prediction'],[case['answer']])['em']
                                    row = {'source_id':source_id,'T':total,'position':position,'repeat':repeat,
                                        'condition':[mode,route,total,condition,trace],
                                        'raw_em':reference_em,'joint_em':em,'raw_prediction':reference['prediction'],
                                        'joint_prediction':generated['prediction'],'answer':case['answer'],
                                        'decisions':serializable(result['decisions'])}
                                    rows.append(row)
                                    append_record(run/'verify/qa.jsonl',row)
                                finally:
                                    runtime.close_context(ctx)
                del raw
    reports = quality_conditions(rows)
    return {'groups':reports,'samples':len(rows),'rows':rows}


def run_calibration(engine, codec, cases, cfg, run, fingerprint, args):
    from .joint_profiles import RuntimeProfile, save_runtime_profile
    costs = collect_costs(engine, codec, cases, cfg, run, state_limit=args.state_limit)
    first = collect_first_costs(engine, cases, cfg, run)
    profile = RuntimeProfile(costs=costs, first_rows=first, fingerprint=fingerprint,
                             ready=False, synthetic=False, text_admitted=True)
    # Measure the actual preloaded selector and CPU control path on cost sources.
    from .joint_runtime import Runtime
    samples = []
    for case in cases if args.state_limit is None else []:
        raw = measure(lambda: engine.prefill(case['context_ids']))[0]
        runtime = Runtime(engine, codec, profile, cfg)
        ctx = runtime.register_context(case['context_ids'], raw, case['case_id'])
        try:
            for repeat in range(cfg['calibration_repeats_per_source']):
                outcome = runtime.run(ctx, case['suffix_ids'], cfg['check_slo_s'], validation_only=True, residual_mode='collect')
                samples.extend(d['select_ns'] for d in outcome['decisions'])
        finally:
            runtime.close_context(ctx)
    if samples:
        profile.select50_ns, profile.select90_ns = (math_ceil(x) for x in summarize(samples))
    save_json(run / 'selector-costs.json', {'samples_ns': samples, 'p50_ns': profile.select50_ns, 'p90_ns': profile.select90_ns})
    save_runtime_profile(run, profile)
    return profile


def math_ceil(value):
    import math
    return math.ceil(value)


def legacy_packets(engine, case, raw, method, whole=False):
    """Build only one baseline's actual original packets; retain old restore path."""
    from .chunks import prepare_chunk
    ids = case['context_ids']
    size = len(ids) if whole else 256
    return [prepare_chunk(engine, ids, raw, method, index, start, min(start+size, len(ids)))
            for index, start in enumerate(range(0, len(ids), size))]


def legacy_run(engine, packets, case, bw_gbps, mode, origin_ns):
    from mini_repro.transport import transfer_chunks
    from .chunks import restore_chunk
    merged = None
    def restore(index, wire):
        nonlocal merged
        merged = restore_chunk(engine, packets[index], wire, merged)
        return merged
    result = transfer_chunks([p.wire for p in packets], bw_gbps, restore,
                             mode='whole' if len(packets) == 1 else 'chunk_' + mode,
                             origin=origin_ns / 1e9, queue_capacity=1)
    token = engine.first([], case['suffix_ids'], merged)
    end = time.perf_counter_ns()
    return {'token': token, 'first_token_ns': end, 't0_ns': origin_ns,
            'ttft_s': (end-origin_ns)/1e9, 'wire_bytes': sum(len(p.wire) for p in packets),
            'network': result['chunks'], 'final_kv': merged}


def benchmark(engine, codec, profile, base, cfg, cases, run, args):
    from .joint_runtime import Runtime
    record_directory = benchmark_directory(run, args.diagnostic_only)
    done = load_benchmark_checkpoint(record_directory, args.diagnostic_only)
    # Stable conditions are the full paired matrix. Step conditions use q2/4K.
    for case in cases:
        total = len(case['context_ids'])
        qualities = [args.fixed_quality] if args.fixed_quality else eligible_qualities(total)
        alternatives = list(groups(qualities))
        if args.group:
            alternatives = [g for g in alternatives if g['name'] in args.group]
            if not alternatives:
                raise ValueError('No requested benchmark group is valid for this context')
        raw, source_s = measure(lambda: engine.prefill(case['context_ids']))
        raw_nll = engine.nll(raw, case['context_ids'], case['continuation_ids'])['nll']
        for mode in ('serial', 'pipeline'):
            local = dict(cfg, execution_mode=mode)
            for bw in cfg['bandwidth_gbps']:
                for slo in cfg['benchmark_slos_seconds']:
                    for condition in cfg.get('benchmark_cache_conditions',cfg['cache_conditions']):
                        for repeat in range(cfg['benchmark_repeats_per_source']):
                            # Rotate group order deterministically to avoid ordering bias.
                            ordered = alternatives[repeat % len(alternatives):] + alternatives[:repeat % len(alternatives)]
                            for group in ordered:
                                if not benchmark_condition_supported(group,mode,condition):
                                    continue
                                key = '/'.join(map(str, (case['case_id'], mode, bw, slo, condition, repeat, group['name'])))
                                if key in done:
                                    continue
                                common = {'key': key, 'source_id': case['source_id'], 'case_id': case['case_id'],
                                    'T': total, 'execution_mode': mode, 'target_gbps': bw, 'slo_s': slo,
                                    'cache_condition': condition, 'repeat': repeat, 'group': group['name'],
                                    'diagnostic_only': args.diagnostic_only, 'source_prefill_s': source_s}
                                if group.get('legacy_adaptive'):
                                    # The old causal controller remains an independent, explicit multi-version baseline.
                                    from .tasks import prepare_chunks, execute_adaptive
                                    legacy_case = dict(case, input_ids=case['context_ids'], answers=[''], answer='')
                                    legacy_cfg = copy.deepcopy(base)
                                    legacy_cfg['adaptive']['length'] = total
                                    started = time.perf_counter_ns()
                                    chunks = prepare_chunks(engine, legacy_cfg, legacy_case, raw)
                                    prep_s = (time.perf_counter_ns()-started)/1e9
                                    result = execute_adaptive(engine, legacy_cfg, legacy_case, chunks,
                                        [bw] * len(chunks), 'adaptive_history', slo,
                                        profile.h_ns(total, len(case['suffix_ids']), .5)/1e9)
                                    if condition == 'cold':
                                        result['ttft_s'] = prep_s + result['ttft_seconds']
                                    else:
                                        result['ttft_s'] = result['ttft_seconds']
                                    result['baseline_preparation_s'] = prep_s
                                    result['baseline_prepared_bytes'] = sum(len(p.wire) for c in chunks for p in c['prepared'].values())
                                elif 'original' in group:
                                    seed_record = None
                                    seed_packets = None
                                    if condition == 'quality_change':
                                        seed_options=seed_request_options(group,condition,total)
                                        seed_start=time.perf_counter_ns()
                                        seed_packets=legacy_packets(engine,case,raw,seed_options['original'])
                                        seed_record=serializable(legacy_run(engine,seed_packets,case,bw,mode,seed_start))
                                        seed_record['seed_options']=seed_options
                                        seed_record['retained_packet_bytes']=sum(len(p.wire) for p in seed_packets)
                                    started = time.perf_counter_ns()
                                    packets = legacy_packets(engine, case, raw, group['original'], group.get('whole', False))
                                    prep_s = (time.perf_counter_ns()-started)/1e9
                                    if condition in ('warm','mixed'):
                                        legacy_run(engine, packets, case, bw, mode, time.perf_counter_ns())
                                        started = time.perf_counter_ns()
                                        if condition == 'mixed':
                                            from .chunks import prepare_chunk
                                            for index in range(0,len(packets),2):
                                                packets[index]=None
                                                begin=index*256
                                                packets[index]=prepare_chunk(engine,case['context_ids'],raw,group['original'],
                                                    index,begin,min(total,begin+256))
                                    result = legacy_run(engine, packets, case, bw, mode, started)
                                    result['baseline_preparation_s'] = prep_s
                                    if seed_record is not None:
                                        result['seed_request']=seed_record
                                else:
                                    runtime = Runtime(engine, codec, profile, local)
                                    ctx = runtime.register_context(case['context_ids'], raw, case['case_id'])
                                    kwargs = {k: group[k] for k in ('encodings', 'fixed_quality', 'route') if k in group}
                                    if 'fixed_quality' in group:
                                        kwargs['quality_mode'] = 'fixed'
                                    try:
                                        seed_record=seed_runtime(runtime,ctx,case,cfg,slo,bw*1e9,group,condition,args)
                                        result = runtime.run(ctx, case['suffix_ids'], slo, bandwidth_bps=bw*1e9,
                                            diagnostic_only=args.diagnostic_only, validation_only=args.diagnostic_only, **kwargs)
                                        if seed_record is not None:
                                            result['seed_request']=seed_record
                                    finally:
                                        runtime.close_context(ctx)
                                if 'final_kv' in result:
                                    result['nll_delta'] = engine.nll(result['final_kv'], case['context_ids'], case['continuation_ids'])['nll'] - raw_nll
                                row = {**common, **serializable(result)}
                                if row.get('seed_request'):
                                    row['seed_and_target_ttft_s']=row['ttft_s']+row['seed_request']['ttft_s']
                                row['slo_met'] = row['ttft_s'] <= slo
                                record_benchmark_completion(record_directory, row, done, args.diagnostic_only)
        if total == 4096 and (args.fixed_quality is None or args.fixed_quality == 2):
            step_benchmark(engine, codec, profile, cfg, case, raw, raw_nll, source_s, run, args, done)
        del raw


def step_benchmark(engine, codec, profile, cfg, case, raw, raw_nll, source_s, run, args, done):
    from .joint_runtime import Runtime
    record_directory = benchmark_directory(run, args.diagnostic_only)
    paired = [g for g in groups([2]) if 'original' not in g and not g.get('legacy_adaptive')
              and g.get('fixed_quality') == 2]
    if args.group:
        paired = [g for g in paired if g['name'] in args.group]
    for mode in ('serial','pipeline'):
        for bw in cfg['bandwidth_gbps']:
            for slo in cfg['benchmark_slos_seconds']:
                for condition in cfg.get('benchmark_cache_conditions',cfg['cache_conditions']):
                    for repeat in range(cfg['benchmark_repeats_per_source']):
                        for group in paired:
                            if not benchmark_condition_supported(group,mode,condition):
                                continue
                            key = '/'.join(map(str, (case['case_id'],'step',mode,bw,slo,condition,repeat,group['name'])))
                            if key in done:
                                continue
                            runtime = Runtime(engine, codec, profile, dict(cfg, execution_mode=mode))
                            ctx = runtime.register_context(case['context_ids'], raw, case['case_id'])
                            kwargs = {'quality_mode': 'fixed', 'fixed_quality': 2,
                                      'diagnostic_only': args.diagnostic_only,
                                      'validation_only': args.diagnostic_only}
                            if 'encodings' in group:
                                kwargs['encodings'] = group['encodings']
                            try:
                                seed_record=seed_runtime(runtime,ctx,case,cfg,slo,bw*1e9,group,condition,args)
                                result = runtime.run(ctx, case['suffix_ids'], slo,
                                    bandwidth_bps=bandwidth_trace('step',bw), **kwargs)
                                if seed_record is not None:
                                    result['seed_request']=seed_record
                                result['nll_delta'] = engine.nll(result['final_kv'], case['context_ids'], case['continuation_ids'])['nll'] - raw_nll
                                row = {**serializable(result), 'key': key, 'source_id': case['source_id'],
                                    'case_id': case['case_id'], 'T': 4096, 'execution_mode': mode,
                                    'target_gbps': bw, 'slo_s': slo, 'cache_condition': condition,
                                    'repeat': repeat, 'group': group['name'], 'trajectory': 'step',
                                    'diagnostic_only': args.diagnostic_only, 'source_prefill_s': source_s}
                                if row.get('seed_request'):
                                    row['seed_and_target_ttft_s']=row['ttft_s']+row['seed_request']['ttft_s']
                                record_benchmark_completion(record_directory, row, done, args.diagnostic_only)
                            finally:
                                runtime.close_context(ctx)


def main(argv=None):
    args = arguments(argv)
    cfg = load_joint_config(args.config)
    from .config import load_config
    base = load_config(repository_path(args.base_config))
    validate_joint_model(base)
    rows, split = read_sources(base, cfg)
    run = run_directory(args.run_dir, run_id=args.run_id)
    summary = plan_summary(cfg, split, run, args)
    if args.plan:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    manifest_path = run / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}
    if args.task == 'inspect' and not args.freeze:
        print(json.dumps({**summary, 'manifest': manifest}, ensure_ascii=False, indent=2))
        return 0
    require_gpu()
    fingerprint, details = code_fingerprint(base, cfg, device=True)
    if manifest and manifest.get('fingerprint') != fingerprint:
        raise ValueError('profile_fingerprint_mismatch: create a new run for changed code/config/environment')
    if args.freeze:
        stage_guard(manifest, 'benchmark')
        profile_for(run, fingerprint, ready=True)
        save_json(run / 'frozen.json', {'fingerprint': fingerprint, 'state': 'frozen',
                  'artifacts': frozen_artifact_digests(run), 'data_sha256':details['data_sha256'],
                  'verified': manifest['verification']})
        manifest['frozen'] = True
        save_json(manifest_path, manifest)
        print(json.dumps({'run_dir': str(run), 'state': 'frozen'}))
        return 0
    task = args.phase if args.task == 'check' else args.task
    if manifest:
        stage_guard(manifest, task, diagnostic_only=args.diagnostic_only)
    elif args.task != 'calibrate':
        raise ValueError('cost_model_missing: calibrate a new run first')
    run.mkdir(parents=True, exist_ok=True)
    if not manifest:
        manifest = {'schema': 'cachegen-joint-v2', 'state': 'initialized', 'fingerprint': fingerprint,
                    'synthetic': False, 'source_split': split, 'gpu_validation': 'external_correctness_gate_required'}
        save_json(manifest_path, manifest)
        save_json(run / 'source-split.json', split)
        save_json(run / 'environment.json', details)
        save_json(run / 'joint-config.json', cfg)
        save_json(run / 'base-config.json', base)
    if task in ('calibrate','fit-residual','verify'):
        invalidate_freeze(run, manifest)
        save_json(manifest_path, manifest)
    if task == 'calibrate':
        gate = run / 'gpu-correctness.json'
        if not gate.exists():
            raise ValueError('GPU correctness evidence missing; run optimization/scripts/server.sh gpu-check first')
        evidence = json.loads(gate.read_text(encoding='utf-8'))
        if not evidence.get('passed') or evidence.get('fingerprint') != fingerprint:
            raise ValueError('GPU correctness evidence invalid or stale')
    from .engine import Engine
    from .joint_codec import JointCodec
    from .joint_profiles import fit_residuals, verify_residuals, save_runtime_profile
    import torch
    torch.set_num_threads(base['cpu_threads'])
    engine = Engine(base['model'], execution=base['execution'])
    codec = JointCodec(engine=engine, device='cuda')
    initialize_codec(codec,startup_encodings(cfg,args),run,task)
    partition = {'calibrate': 'cost', 'fit-residual': 'residual', 'verify': 'verification', 'benchmark': 'test'}[task]
    cases = tokenize_cases(rows, split[partition], engine.tokenizer, cfg['context_lengths'], base['quality']['continuation'])
    save_json(run / f'{task}-context_manifest.json', [manifest_for(c, eligible_qualities(len(c['context_ids'])), cfg['chunk_tokens']) for c in cases])
    try:
        if task == 'calibrate':
            run_calibration(engine, codec, cases, cfg, run, fingerprint, args)
            manifest['state'] = 'diagnostic_costs' if args.state_limit else 'costs'
            manifest['calibration_complete'] = args.state_limit is None
        elif task in ('fit-residual', 'verify'):
            profile = profile_for(run, fingerprint)
            _, records = check_requests(engine, codec, profile, cfg, cases, run, task)
            if task == 'fit-residual':
                profile.residual_table = fit_residuals(records)
                save_json(run / 'residual-profile.json', profile.residual_table.to_dict())
                manifest['state'] = 'residual'
            else:
                qa = qa_acceptance(engine, codec, profile, cfg, split['verification'], run)
                qa_map = {tuple(group['condition']):group['em_retention'] for group in qa['groups']}
                for record in records:
                    record['em_retention'] = qa_map.get(tuple(record['quality_condition']),0.)
                required = [(m, r, t, c) for m in ('serial','pipeline') for t in cfg['context_lengths']
                            for r in ('chunks','whole_text') for c in cfg['cache_conditions']
                            if not (r == 'whole_text' and c == 'mixed')]
                report = verify_residuals(records, profile.residual_table, required_groups=required,
                                          required_bandwidth_classes=cfg['bandwidth_trajectories'])
                if not qa['groups'] or not all(g['passed'] for g in qa['groups']):
                    report['ready'] = False
                    report['quality_ok'] = False
                report['qa'] = {k:v for k,v in qa.items() if k != 'rows'}
                profile.ready = report['ready']
                save_json(run / 'verification.json', report)
                manifest.update(state='ready' if profile.ready else 'failed_verification', verification=report)
            save_runtime_profile(run, profile)
        elif task == 'benchmark':
            if not args.diagnostic_only:
                verify_frozen_artifacts(run,fingerprint,manifest)
            profile = profile_for(run, fingerprint, ready=not args.diagnostic_only)
            benchmark(engine, codec, profile, base, cfg, cases, run, args)
            manifest['diagnostic_benchmark_complete' if args.diagnostic_only else 'benchmark_complete'] = True
    except Exception as exc:
        if task != 'benchmark':
            invalidate_freeze(run, manifest)
            manifest['state'] = 'failed_' + task
        else:
            manifest['diagnostic_benchmark_complete' if args.diagnostic_only else 'benchmark_complete'] = False
        manifest['failure'] = {'type':type(exc).__name__,'message':str(exc),
                              'joint_failure':serializable(getattr(exc,'joint_failure',None))}
        save_json(run / ('failure-' + task + '.json'), manifest['failure'])
        save_json(manifest_path, manifest)
        raise
    save_json(manifest_path, manifest)
    print(json.dumps({'run_dir': str(run), 'state': manifest['state']}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f'joint: {exc}', file=sys.stderr)
        raise SystemExit(2)
