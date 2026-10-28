"""Opt-in bridge to existing preparation, socket transport and TTFT replay."""
import random
import statistics
import time
import torch


class CodecEngine:
    """Existing Engine API, extended without changing historical method meanings."""
    def __init__(self, engine, codecs):
        self.base, self.codecs = engine, codecs

    def __getattr__(self, name):
        return getattr(self.base, name)

    def encode(self, blob, method):
        from experiments.engine import timed
        if method not in self.codecs:
            return self.base.encode(blob, method)
        codec = self.codecs[method]
        (payload, _), seconds = timed(lambda: codec.encode(blob))
        return payload, seconds, codec.decode

    def restore_function(self, method, tokens):
        return self.codecs[method].decode if method in self.codecs else self.base.restore_function(method, tokens)


def adaptive_replay(engine, cfg, packets, costs, trace, context, suffix, quality_order,
                    first_seconds, strategy, fixed_method, highest_method, slo):
    from experiments.engine import timed
    from experiments.chunks import restore_chunk
    from mini_repro.transport import transfer_chunks
    from .policy import choose, fixed_strategy_method
    p = cfg['network']
    merged = None
    torch.cuda.synchronize()
    start = time.perf_counter()
    # Probe traffic and setup are inside TTFT and byte accounting for every strategy.
    probe_payload = bytes(p['probe_bytes'])
    probe_result = transfer_chunks([probe_payload], trace[0], lambda _, wire: None,
                                   mode='whole', chunk_bytes=p['pacing_quantum_bytes'], origin=start)
    if probe_result['received'] != [probe_payload]:
        raise ValueError('Probe byte identity failed')
    probe = {'bytes': len(probe_payload), 'seconds': probe_result['chunks'][0]['effective_transfer_seconds']}
    completed, decisions = [], []
    received_bytes = len(probe_payload)
    for index in range(len(costs)):
        decision = choose(costs, index, probe, completed, slo, time.perf_counter() - start,
                          first_seconds, p['safety_fraction'], p['min_reserve_seconds'], quality_order)
        override = fixed_strategy_method(strategy, fixed_method, highest_method)
        if override is not None:
            decision['method'] = override
            decision['forecast_feasible'] = decision['candidate_forecasts_seconds'][decision['method']] <= decision['remaining_budget_seconds']
        method = decision['method']
        packet = packets[method][index]
        def restore(_, wire):
            nonlocal merged
            merged = restore_chunk(engine, packet, wire, merged)
            torch.cuda.synchronize()
        # Only transport sees this trace element; policy has only completed observations.
        result = transfer_chunks([packet.wire], trace[index], restore, mode='whole',
                                 chunk_bytes=p['pacing_quantum_bytes'], origin=start)
        if result['received'] != [packet.wire]:
            raise ValueError('Adaptive transport changed bytes')
        network = result['chunks'][0]
        completed.append({'bytes': len(packet.wire), 'seconds': network['effective_transfer_seconds']})
        received_bytes += len(packet.wire)
        decisions.append({**decision, 'actual_transport_gbps': trace[index], 'network': network})
    token, first_s = timed(lambda: engine.first(context, suffix, merged))
    wall = time.perf_counter() - start
    return merged, {'strategy': strategy, 'ttft_seconds': wall, 'slo_seconds': slo,
                    'slo_violation': wall > slo, 'probe': probe,
                    'total_bytes_including_probe': received_bytes, 'decisions': decisions,
                    'first_token_id': token, 'remaining_input_seconds': first_s,
                    'first_token_reserve_seconds': first_seconds,
                    'infeasible_decisions': sum(not d['forecast_feasible'] for d in decisions),
                    'scheduling_mode': 'serial_chunk_decisions', 'probe_cost_included': True}


def run(args, cfg):
    from tools.paper_study import run_folder, selection, load_profiles, inputs, qa_cases, score_answer, records
    from tools.paper_compression import setup, save, append
    from experiments.chunks import prepare, restore_chunk, preparation_metadata
    from experiments.engine import timed
    from experiments.ttft import replay
    from .codec import PaperCodec
    from .study_config import candidates
    if args.backend != 'cuda':
        raise ValueError('Performance measurements require real CUDA encoding/decoding')
    frozen = selection(args, cfg)
    core = frozen['selected']['core']
    options = candidates(cfg)
    reference = options[core].anchor_reference
    names = [core] if args.action == 'ttft' else [n for n, c in options.items()
             if n.startswith('core-') and c.anchor_reference == reference]
    with run_folder(args, cfg) as data_hash:
        engine, identity = setup(cfg)
        profiles = load_profiles(args.profiles, cfg, identity, data_hash, names)
        if profiles[core].identity != frozen['profile_identities'][core]:
            raise ValueError('Network codec differs from validation-selected codec')
        engine.load()
        qa = qa_cases(engine, cfg)
        adapter = CodecEngine(engine, {n: PaperCodec(p, identity, 'cuda') for n, p in profiles.items()})
        p = cfg['network']
        all_cases = inputs(engine, cfg, 'test', [p['tokens']])
        cases = [c for c in all_cases if c['source_id'] in p['source_ids']]
        save(args.output / 'selection.json', cases)
        save(args.output / 'model_identity.json', identity)
        save(args.output / 'frozen_selection.json', frozen)
        transport_cfg = {'transport': {'queue_capacity': p['queue_capacity'],
                         'pacing_quantum_bytes': p['pacing_quantum_bytes']}}
        validation_quality = {r['method']: statistics.mean(r['mean_delta_nll_by_length'].values())
                              for r in frozen['candidates']}
        quality_order = sorted(names, key=lambda n: (validation_quality[n], n))
        highest_method = max(names, key=lambda n: options[n].vector_range)
        for case in cases:
            context, continuation = case['context_ids'], case['continuation_ids']
            suffix = qa[case['source_id']]['suffix_ids']
            blob = engine.prefill(context)
            raw_nll = engine.nll(blob, context, continuation)['nll']
            methods = names + (['uniform8', 'uniform4', 'text'] if args.action == 'ttft' else ['text'])
            sizes = [len(context), p['chunk_tokens']] if args.action == 'ttft' else [p['chunk_tokens']]
            packets = {(size, m): prepare(adapter, context, blob, m, size) for size in sizes for m in methods}
            quality = {}
            # Explicit restoration parity check, outside measured TTFT.
            for size in sizes:
                for method in methods:
                    serial, _ = replay(adapter, transport_cfg, packets[size, method], None, 'chunk_serial', context, suffix)
                    pipeline, _ = replay(adapter, transport_cfg, packets[size, method], None, 'chunk_pipeline', context, suffix)
                    if not torch.equal(serial, pipeline):
                        raise ValueError('Same prepared bytes restored differently across transport modes')
                    quality[size, method] = {'delta_nll': engine.nll(serial, context, continuation)['nll'] - raw_nll,
                                             'qa': score_answer(engine, serial, qa[case['source_id']])}
                    append(args.output / 'preparation.jsonl', {'source_id': case['source_id'], 'method': method,
                           'chunk_tokens': size, 'mode_kv_identity': True,
                           **preparation_metadata(packets[size, method])})
                    del serial, pipeline
            if args.action == 'ttft':
                schedule = [(m, mode, bw, repeat) for m in methods for mode in ('whole', 'chunk_serial', 'chunk_pipeline')
                            for bw in p['bandwidth_gbps'] for repeat in range(p['repeats'] + 1)]
                random.Random(cfg['seed'] + case['source_id']).shuffle(schedule)
                schedule.sort(key=lambda step: step[3] > 0)  # Finish all warmups before measurements.
                for method, mode, bw, repeat in schedule:
                    size = len(context) if mode == 'whole' else p['chunk_tokens']
                    merged, row = replay(adapter, transport_cfg, packets[size, method], bw, mode, context, suffix)
                    if repeat:  # repeat=0 warmup; all methods/modes/bandwidths have one.
                        row.update(source_id=case['source_id'], method=method, repetition=repeat,
                                   target_bandwidth_gbps=bw, phase='measure', **quality[size, method],
                                   quality_scope='once_per_source_method_chunk_geometry_reused_for_timing_repeats')
                        append(args.output / 'records.jsonl', row)
                    del merged
            else:
                costs = []
                for index in range(len(packets[p['chunk_tokens'], core])):
                    item = {}
                    for method in methods:
                        packet = packets[p['chunk_tokens'], method][index]
                        previous = blob[:, :, :, :packet.start, :] if packet.start else None
                        times = []
                        for _ in range(3):
                            restored, seconds = timed(lambda: restore_chunk(adapter, packet, packet.wire, previous))
                            times.append(seconds)
                            del restored
                        item[method] = {'bytes': len(packet.wire), 'restore_seconds': max(times)}
                    costs.append(item)
                reserve_times = []
                for _ in range(3):
                    _, seconds = timed(lambda: engine.first(context, suffix, blob))
                    reserve_times.append(seconds)
                save(args.output / f'costs-{case["source_id"]}.json', {'costs': costs, 'first_seconds': reserve_times,
                     'quality_order_from_validation': quality_order, 'fixed_selected_method': core,
                     'fixed_highest_quality_method': highest_method,
                     'highest_quality_definition': 'largest vector_range, selected anchor reference'})
                prepared = {m: packets[p['chunk_tokens'], m] for m in methods}
                schedule = [(trace_id, slo, repeat, strategy)
                            for trace_id in range(len(p['traces_gbps'])) for slo in p['slo_seconds']
                            for repeat in range(p['repeats'])
                            for strategy in ('history', 'fixed', 'fixed_highest_quality', 'text')]
                random.Random(cfg['seed'] + case['source_id']).shuffle(schedule)
                for trace_id, slo, repeat, strategy in schedule:
                    trace = p['traces_gbps'][trace_id]
                    if len(trace) != len(costs):
                        raise ValueError('Trace length must equal chunk count')
                    merged, row = adaptive_replay(adapter, cfg, prepared, costs, trace, context, suffix,
                        quality_order, max(reserve_times), strategy, core, highest_method, slo)
                    row.update(source_id=case['source_id'], trace_id=trace_id, repetition=repeat,
                               delta_nll=engine.nll(merged, context, continuation)['nll'] - raw_nll,
                               qa=score_answer(engine, merged, qa[case['source_id']]))
                    append(args.output / 'records.jsonl', row)
                    del merged
            del blob, packets
            torch.cuda.empty_cache()
        rows = records(args.output / 'records.jsonl')
        groups = {}
        for row in rows:
            key = ((row['method'], row['mode'], row['target_bandwidth_gbps']) if args.action == 'ttft'
                   else (row['strategy'], row['trace_id'], row['slo_seconds']))
            groups.setdefault(key, []).append(row)
        summary = []
        for key, group in groups.items():
            times = [r['ttft_wall_seconds'] if args.action == 'ttft' else r['ttft_seconds'] for r in group]
            entry = {'condition': list(key), 'timing_runs': len(group),
                     'independent_sources': len({r['source_id'] for r in group}),
                     'mean_ttft_seconds': statistics.mean(times), 'median_ttft_seconds': statistics.median(times),
                     'mean_delta_nll': statistics.mean(r['delta_nll'] for r in group),
                     'mean_qa_f1': statistics.mean(r['qa']['f1'] for r in group),
                     'mean_qa_em': statistics.mean(r['qa']['em'] for r in group)}
            if args.action == 'adaptive':
                entry.update(slo_violation_runs=sum(r['slo_violation'] for r in group),
                             mean_bytes_with_probe=statistics.mean(r['total_bytes_including_probe'] for r in group))
            summary.append(entry)
        save(args.output / 'summary.json', {'rows': summary, 'repeats_are_independent_qa_samples': False,
             'scope': 'single-GPU local paced socket; prepositioned probability models',
             'parameter_status': cfg['parameter_status']})
