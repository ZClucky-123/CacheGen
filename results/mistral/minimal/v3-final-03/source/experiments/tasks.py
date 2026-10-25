"""Shared quality/adaptive tasks and TTFT/ablation exports for every adapter."""
import json
import statistics
import time
import torch
from experiments.engine import timed, validate_blob
from experiments.records import save, adaptive_key
from experiments.scoring import extract_answer, score_qa, phrase_match
from mini_repro.adaptive import next_decision
from mini_repro.ttft import measurement_schedule
from mini_repro.transport import transfer_chunks
from experiments.chunks import prepare_chunk, unframe, restore_chunk
from experiments.feedback import CompletedTransfer, history_decision


def case_fields(case):
    return {k: v for k, v in case.items() if k not in ('context_ids', 'input_ids', 'continuation_ids', 'suffix_ids')}


def event(run, **row):
    with (run / 'events.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def calibrate(cfg, run):
    rows = []
    for size in cfg['calibration']['payload_bytes']:
        payload = bytes(range(256)) * (size // 256) + bytes(range(size % 256))
        for bw in [None] + cfg['ttft']['bandwidth_gbps']:
            transfer_chunks([payload], bw, lambda _, data: data, mode='whole',
                            chunk_bytes=cfg['transport']['pacing_quantum_bytes'])
            for repeat in range(cfg['calibration']['repeats']):
                result = transfer_chunks([payload], bw, lambda _, data: data, mode='whole',
                                         chunk_bytes=cfg['transport']['pacing_quantum_bytes'])
                if result['value'] != payload:
                    raise RuntimeError('Calibration transport mismatch')
                network = result['chunks'][0]
                seconds = network['effective_transfer_seconds']
                realized = size * 8 / seconds / 1e9
                rows.append({'payload_bytes': size, 'target_gbps': bw, 'repeat': repeat,
                             'seconds': seconds, 'actual_gbps': realized,
                             'transport_wall_seconds': result['transport_wall_seconds'],
                             'receiver_data_available': network['data_available'],
                             'timeline': network, 'pacing_quantum_bytes': result['pacing_quantum_bytes'],
                             'actual_to_target': realized / bw if bw else None,
                             'target_reached_90pct': realized >= bw * .9 if bw else None})
                save(run / 'calibration.json', {'mode': 'local_socket_pre_send_budget',
                    'boundary': 'receiver buffer ready to complete bytes available; includes pre-send pacing and socket/materialization, excludes setup and capacity wait; wall/setup separately recorded',
                    'rows': rows})


def cache_checks(engine, cfg, selected, run):
    rows = []
    def full_generate(ids, limit):
        expected = engine.model.generate(torch.tensor([ids], device='cuda'),
            attention_mask=torch.ones((1, len(ids)), dtype=torch.long, device='cuda'),
            max_new_tokens=limit, do_sample=False, pad_token_id=engine.tokenizer.eos_token_id)[0, len(ids):].tolist()
        return expected[:expected.index(engine.tokenizer.eos_token_id)] if engine.tokenizer.eos_token_id in expected else expected
    # Multiple independent real documents at all configured lengths, multistep generation.
    candidates = selected['ttft']
    for case in candidates:
        context, suffix = case['context_ids'], case['suffix_ids']
        blob = engine.prefill(context)
        before = blob.clone()
        full = engine.model(input_ids=torch.tensor([context + suffix], device='cuda'), use_cache=True)
        direct = full.logits[0, -1].float()
        optimized, _ = engine.forward(context + suffix)
        reused, _ = engine.forward(suffix, blob)
        error = float((direct - reused[0, -1]).abs().max())
        optimized_error = float((direct - optimized[0, -1]).abs().max())
        if not torch.allclose(direct, optimized[0, -1], atol=.1, rtol=.02) or int(direct.argmax()) != int(optimized[0, -1].argmax()):
            raise RuntimeError(f'Optimized text logits differ: {case["case_id"]}')
        if not torch.allclose(direct, reused[0, -1], atol=.1, rtol=.02) or int(direct.argmax()) != int(reused[0, -1].argmax()):
            raise RuntimeError(f'Raw cache logits differ: {case["case_id"]}')
        del full, direct, reused, optimized
        limit = cfg['checks']['generation_tokens']
        expected = full_generate(context + suffix, limit)
        generated = engine.generate(suffix, blob, limit)
        repeated = engine.generate(suffix, blob, limit)
        if expected != generated['generated_ids'] or expected != repeated['generated_ids'] or not torch.equal(blob, before):
            raise RuntimeError(f'Multitoken cache generation or mutation check failed: {case["case_id"]}')
        natural_suffix = case['continuation_ids']
        natural_expected = full_generate(context + natural_suffix, limit)
        natural_reused = engine.generate(natural_suffix, blob, limit)['generated_ids']
        if natural_expected != natural_reused:
            raise RuntimeError(f'Natural multistep cache generation differs: {case["case_id"]}')
        # Full-context cache sliced into raw chunks must reproduce all tokens, including a tail.
        chunk_size = cfg['adaptive']['chunk_size']
        slices = [blob[:, :, :, start:start + chunk_size, :] for start in range(0, len(context), chunk_size)]
        merged = torch.cat(slices, dim=3)
        if not torch.equal(merged, blob) or engine.generate(suffix, merged, limit)['generated_ids'] != expected:
            raise RuntimeError('Chunk assembly lost positions or tail')
        # Text recomputation must depend on prior chunks, rather than independent chunk prefills.
        text_merged = None
        for start in range(0, len(context), chunk_size):
            text_merged = engine.prefill(context[start:start + chunk_size], text_merged)
        if engine.generate(suffix, text_merged, limit)['generated_ids'] != expected:
            raise RuntimeError('Sequential text chunk cache differs from full prefill')
        rows.append({'case_id': case['case_id'], 'context_tokens': len(context), 'suffix_tokens': len(suffix),
            'kv_shape': list(blob.shape), 'raw_logits_max_error': error, 'optimized_logits_max_error': optimized_error,
            'generated_ids': expected, 'natural_suffix_ids': natural_suffix, 'natural_generated_ids': natural_expected,
            'repeat_cache_unchanged': True, 'raw_chunks_and_text_chunks_match': True})
        save(run / 'cache_checks.json', {'status': 'running', 'tolerances': {'atol': .1, 'rtol': .02}, 'cases': rows})
        del before, blob, merged, text_merged, slices
        torch.cuda.empty_cache()
    # Check complete continuation scoring against the unmodified model's teacher forcing.
    nll_rows = []
    for case in selected['quality'][:2]:
        ids, continuation = case['context_ids'], case['continuation_ids']
        blob = engine.prefill(ids)
        result = engine.nll(blob, ids, continuation)
        logits = engine.model(input_ids=torch.tensor([ids + continuation[:-1]], device='cuda'), use_cache=False).logits
        losses = torch.nn.functional.cross_entropy(logits[0, len(ids)-1:].float(), torch.tensor(continuation, device='cuda'), reduction='none')
        delta = abs(result['nll'] - float(losses.mean()))
        if delta > .01:
            raise RuntimeError(f'Full-continuation NLL mismatch: {delta}')
        nll_rows.append({'case_id': case['case_id'], 'tokens_scored': len(continuation),
                         'full_nll': float(losses.mean()), 'reuse_nll': result['nll'], 'absolute_error': delta})
        del blob, logits, losses
    save(run / 'cache_checks.json', {'status': 'passed', 'tolerances': {'atol': .1, 'rtol': .02, 'nll_abs': .01},
                                    'cases': rows, 'complete_nll_checks': nll_rows})


def quality(engine, cfg, selected, run, journal, job, progress):
    task = job['name']
    for kind in ('quality', 'controlled', 'triviaqa'):
        for case in selected[kind]:
            missing = [m for m in cfg['quality']['methods'] if f'{task}/{kind}/{case["case_id"]}/{m}' not in journal.keys]
            if not missing:
                continue
            context = case['context_ids'] if kind == 'quality' else case['input_ids'][:-1] if kind == 'triviaqa' else case['input_ids']
            suffix = case['input_ids'][-1:] if kind == 'triviaqa' else case.get('suffix_ids')
            progress(task, f'{task}/{kind}/{case["case_id"]}/offline-prefill')
            blob, prefill_s = timed(lambda: engine.prefill(context))
            for method in missing:
                key = f'{task}/{kind}/{case["case_id"]}/{method}'
                progress(task, key)
                payload, encode_s, restore = engine.encode(blob, method)
                restored, decode_s = timed(lambda: restore(payload))
                validate_blob(restored, blob)
                row = {**case_fields(case), 'kind': kind, 'method': method,
                       'cached_tokens': len(context), 'raw_tensor_bytes': blob.numel() * blob.element_size(),
                       'offline_prefill_seconds': prefill_s, 'offline_prefill_scope': 'once_per_case_annotation',
                       'bytes': len(payload), 'offline_encode_seconds': encode_s, 'decode_seconds': decode_s,
                       'kv_mae': float((blob - restored).abs().float().mean())}
                if kind == 'quality':
                    row.update(engine.nll(restored, context, case['continuation_ids']))
                else:
                    limit = cfg['triviaqa' if kind == 'triviaqa' else 'quality']['max_new_tokens']
                    row.update(engine.generate(suffix, restored, limit))
                    row.update(score_qa(row['prediction'], case['answers']))
                    row['scored_answer'] = extract_answer(row['prediction'])
                    if kind == 'controlled':
                        row['historical_phrase_match'] = phrase_match(row['prediction'], case['answer'])
                journal.add(key, task, row)
                del restored, payload, restore
            del blob
            torch.cuda.empty_cache()


# Same TTFT implementation for every adapter.
from experiments.ttft import ttft


def prepare_chunks(engine, cfg, case, blob):
    p = cfg['adaptive']
    chunks = []
    ids = case['input_ids']
    for index, start in enumerate(range(0, len(ids), p['chunk_size'])):
        end = min(start + p['chunk_size'], len(ids))
        previous = blob[:, :, :, :start, :] if start else None
        prepared = {}
        row = {'start': start, 'end': end, 'levels': {}, 'prepared': prepared}
        for level in (0, 1, 2, 3):
            packet = prepare_chunk(engine, ids, blob, 'text' if level == 0 else f'cachegen{level}', index, start, end)
            prepared[level] = packet
            def work():
                return restore_chunk(engine, packet, packet.wire, previous)
            for _ in range(p['warmup']):
                work()
            timings = []
            for _ in range(p['repeats']):
                restored, elapsed = timed(work)
                timings.append(elapsed)
                if restored.shape[3] != end or not bool(torch.isfinite(restored).all()):
                    raise RuntimeError('Adaptive preparation shape or finite check failed')
                del restored
            cost = {'bytes': len(packet.wire), 'calibration_seconds': timings,
                    'offline_encode_seconds': packet.encode_seconds,
                    'message_sha256': packet.sha256, **packet.metadata}
            # Wire byte count includes frame; payload_bytes remains separately named.
            cost['bytes'] = len(packet.wire)
            if level == 0:
                cost.update(prefill_seconds=statistics.median(timings),
                            **engine.text_metadata(unframe(packet.wire)[1]))
                row['text'] = cost
            else:
                cost['decode_seconds'] = statistics.median(timings)
                row['levels'][level] = cost
        chunks.append(row)
    return chunks


def execute_adaptive(engine, cfg, case, chunks, trace, strategy, slo, reserve):
    merged = None
    decisions = []
    completed = []
    torch.cuda.synchronize()
    start = time.perf_counter()
    for index, chunk in enumerate(chunks):
        decision_start = time.perf_counter()
        elapsed = decision_start - start
        if strategy == 'adaptive_history':
            decision = history_decision(chunks, index, tuple(completed),
                cfg['adaptive']['initial_bandwidth_gbps'], slo, elapsed, reserve)
        else:
            decision = next_decision(chunks, index, trace[index], slo, elapsed, strategy, reserve)
            decision.update(estimate_gbps=trace[index], estimate_source='current_trace', source_chunk_index=None)
        decision.update(decision_start=decision_start - start,
                        decision_seconds=time.perf_counter() - decision_start)
        level = decision['level']
        packet = chunk['prepared'][level]
        def reconstruct(_, received):
            nonlocal merged
            merged = restore_chunk(engine, packet, received, merged)
            torch.cuda.synchronize()
            return merged
        # Only the executor reads the current trace for the history variant.
        result = transfer_chunks([packet.wire], trace[index], reconstruct, mode='whole',
            queue_capacity=1, chunk_bytes=cfg['transport']['pacing_quantum_bytes'], origin=start)
        network = result['chunks'][0]
        transfer_s = network['effective_transfer_seconds']
        completed.append(CompletedTransfer(index, len(packet.wire), transfer_s))
        decision.update(start=chunk['start'], end=chunk['end'], cache_tokens_after=merged.shape[3],
                        bytes=len(packet.wire), message_sha256=packet.sha256,
                        transfer_seconds=transfer_s, effective_transfer_seconds=transfer_s,
                        reconstruct_seconds=network['restore_seconds'], network_timeline=network,
                        configured_bandwidth_gbps=trace[index], actual_bandwidth_gbps=network['actual_bandwidth_gbps'],
                        fallback_to_text=level == 0, byte_identity_checked=result['received'] == [packet.wire])
        if not decision['byte_identity_checked']:
            raise RuntimeError('Adaptive received bytes differ')
        if level == 0:
            decision.update(engine.text_metadata(unframe(packet.wire)[1]))
        decisions.append(decision)
    initial, first_s = timed(lambda: engine.first_state([], case['suffix_ids'], merged))
    wall = time.perf_counter() - start
    generated = engine.generate(case['suffix_ids'], merged, cfg['adaptive']['max_new_tokens'], initial=initial)
    torch.cuda.synchronize()
    answer_wall = time.perf_counter() - start
    if not bool(torch.isfinite(merged).all()):
        raise RuntimeError('Adaptive merged cache contains nonfinite values')
    return {**generated, **score_qa(generated['prediction'], case['answers']),
           'scored_answer': extract_answer(generated['prediction']), 'decisions': decisions,
           'levels': [d['level'] for d in decisions], 'text_fallback_chunks': sum(d['level'] == 0 for d in decisions),
           'mode': 'socket_serial', 'throughput_estimator': cfg['adaptive']['throughput_estimator'] if strategy == 'adaptive_history' else 'current_trace',
           'initial_bandwidth_gbps': cfg['adaptive']['initial_bandwidth_gbps'], 'chunk_size': cfg['adaptive']['chunk_size'],
           'transfer_seconds': sum(d['transfer_seconds'] for d in decisions),
           'reconstruct_seconds': sum(d['reconstruct_seconds'] for d in decisions),
           'remaining_input_seconds': first_s, 'local_wall_seconds': wall, 'ttft_seconds': wall,
           'measured_ttft_seconds': wall, 'answer_latency_seconds': answer_wall,
           'slo_seconds': slo, 'slo_violation': wall > slo, 'first_token_reserve_seconds': reserve,
           'estimated_network_seconds': 0.,
           'historical_phrase_match': phrase_match(generated['prediction'], case['answer'])}


def adaptive(engine, cfg, selected, run, journal, job, progress):
    p, task = cfg['adaptive'], job['name']
    for case_index, case in enumerate(selected['adaptive']):
        prefix = f'{task}/{case["case_id"]}/'
        expected = p['traces'] * p['repeats'] * len(p['strategies'])
        if sum(k.startswith(prefix) for k in journal.keys) == expected:
            continue
        progress(task, f'{prefix}offline-preparation')
        preparation_start = time.perf_counter()
        blob, prefill_s = timed(lambda: engine.prefill(case['input_ids']))
        raw = engine.generate(case['suffix_ids'], blob, p['max_new_tokens'])
        raw_score = score_qa(raw['prediction'], case['answers'])
        for _ in range(p['warmup']):
            engine.first([], case['suffix_ids'], blob)
        first_times = [timed(lambda: engine.first([], case['suffix_ids'], blob))[1] for _ in range(p['repeats'])]
        reserve = statistics.median(first_times)
        chunks = prepare_chunks(engine, cfg, case, blob)
        preparation_s = time.perf_counter() - preparation_start
        all_encodes_s = sum(r['offline_encode_seconds'] for c in chunks for r in c['levels'].values())
        event(run, event='adaptive_calibration', task=task, case_id=case['case_id'],
              chunks=[{k: c[k] for k in ('start', 'end', 'text', 'levels')} for c in chunks], first_token_seconds=first_times,
              offline_prefill_seconds=prefill_s, offline_all_levels_encode_seconds=all_encodes_s,
              offline_preparation_wall_seconds=preparation_s,
              preparation_boundary='full prefill, raw QA, warmups, encode/decode calibration and validation; not minimal cache build cost')
        del blob
        for trace_id, raw_trace in enumerate(selected['traces']):
            trace = [b * job['scale'] for b in raw_trace]
            schedule = measurement_schedule(p['strategies'], [1], warmup=p['warmup'], repeats=p['repeats'],
                                             seed=cfg['seed'] + case_index * 100 + trace_id)
            event(run, event='adaptive_schedule', task=task, case_id=case['case_id'], trace_id=trace_id, schedule=schedule)
            for step_index, step in enumerate(schedule):
                strategy = step['method']
                key = adaptive_key(task, case['case_id'], trace_id, step['repetition'], strategy, cfg)
                if step['phase'] == 'measure' and key in journal.keys:
                    continue
                progress(task, key)
                row = execute_adaptive(engine, cfg, case, chunks, trace, strategy, job['slo'], reserve)
                row.update(case_fields(case), kind='adaptive', strategy=strategy, trace_id=trace_id,
                    repetition=step['repetition'], phase=step['phase'], method_order=step['method_order'],
                    schedule_position=step_index, bandwidth_trace_gbps=trace, bandwidth_scale=job['scale'],
                    raw_prediction=raw['prediction'], raw_em=raw_score['em'], raw_f1=raw_score['f1'])
                row.update(offline_prefill_seconds=prefill_s, offline_all_levels_encode_seconds=all_encodes_s,
                           offline_preparation_wall_seconds=preparation_s, offline_preparation_scope='once_per_task_fact_annotation')
                if step['phase'] == 'measure':
                    journal.add(key, task, row)
                else:
                    event(run, event='warmup', task=task, observation=row)
        del chunks
        torch.cuda.empty_cache()


from experiments.ablation import ablation
