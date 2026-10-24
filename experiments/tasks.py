"""The same three experiment implementations run against every model adapter."""
import json
import random
import statistics
import time
import torch
from experiments.engine import timed, validate_blob
from experiments.records import save
from experiments.scoring import extract_answer, score_qa, phrase_match
from mini_repro.adaptive import next_decision, representation_cost
from mini_repro.ttft import measurement_schedule
from mini_repro.transport import rate_limited_transfer


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
            rate_limited_transfer(payload, bw, validate=True)
            for repeat in range(cfg['calibration']['repeats']):
                received, seconds = rate_limited_transfer(payload, bw, validate=True)
                if received != payload:
                    raise RuntimeError('Calibration transport mismatch')
                realized = size * 8 / seconds / 1e9
                rows.append({'payload_bytes': size, 'target_gbps': bw, 'repeat': repeat,
                             'seconds': seconds, 'actual_gbps': realized,
                             'actual_to_target': realized / bw if bw else None,
                             'target_reached_90pct': realized >= bw * .9 if bw else None})
                save(run / 'calibration.json', {'mode': 'local_socket_serial',
                    'boundary': 'socket setup + sender/receiver + pacing + bytes materialization; debug equality outside transfer clock',
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


def ttft(engine, cfg, selected, run, journal, job, progress):
    task, p = job['name'], cfg['ttft']
    for index, case in enumerate(selected['ttft']):
        prefix = f'{task}/{case["case_id"]}/'
        expected = len(p['methods']) * len(p['bandwidth_gbps']) * p['repeats']
        if sum(k.startswith(prefix) for k in journal.keys) == expected:
            continue
        context, suffix = case['context_ids'], case['suffix_ids']
        progress(task, f'{prefix}offline-prefill')
        blob, prefill_s = timed(lambda: engine.prefill(context))
        direct = engine.first(context, suffix)
        if engine.first(context, suffix, blob) != direct:
            raise RuntimeError('TTFT raw cache parity failed')
        payloads = {}
        for method in p['methods']:
            if method == 'text':
                payloads[method] = (engine.text_payload(context), 0.0, engine.restore_text)
            else:
                payloads[method] = engine.encode(blob, method)
                restored = payloads[method][2](payloads[method][0])
                validate_blob(restored, blob)
                del restored
        del blob
        schedule = measurement_schedule(p['methods'], p['bandwidth_gbps'], warmup=p['warmup'],
                                         repeats=p['repeats'], seed=cfg['seed'] + index)
        event(run, event='ttft_schedule', task=task, case_id=case['case_id'], schedule=schedule)
        for step_index, step in enumerate(schedule):
            key = f'{prefix}{step["bandwidth_gbps"]}/{step["repetition"]}/{step["method"]}'
            if step['phase'] == 'measure' and key in journal.keys:
                continue
            progress(task, key)
            payload, encode_s, restore = payloads[step['method']]
            torch.cuda.synchronize()
            start = time.perf_counter()
            received, transfer_s = rate_limited_transfer(payload, step['bandwidth_gbps'])
            restored, decode_s = timed(lambda: restore(received))
            if step['method'] == 'text':
                token, first_s = timed(lambda: engine.first(restored, suffix))
            else:
                token, first_s = timed(lambda: engine.first(context, suffix, restored))
            wall = time.perf_counter() - start
            # Debug checks outside the measured window.
            if received != payload or (step['method'] == 'text' and restored != context):
                raise RuntimeError('TTFT transport/token identity failed')
            row = {**case_fields(case), 'kind': 'ttft', **step, 'schedule_position': step_index,
                'bytes': len(payload), 'offline_encode_seconds': encode_s, 'transfer_seconds': transfer_s,
                'offline_prefill_seconds': prefill_s, 'offline_prefill_scope': 'once_per_context_annotation',
                'reconstruct_seconds': decode_s, 'remaining_input_seconds': first_s, 'ttft_wall_seconds': wall,
                'actual_bandwidth_gbps': len(payload) * 8 / transfer_s / 1e9,
                'actual_to_target': len(payload) * 8 / transfer_s / 1e9 / step['bandwidth_gbps'],
                'first_token_id': token, 'matches_raw': token == direct, 'mode': 'socket_receive_then_restore'}
            if step['method'] == 'text':
                row.update(engine.text_metadata(payload))
            if step['phase'] == 'measure':
                journal.add(key, task, row)
            else:
                event(run, event='warmup', task=task, observation=row)
            del received, restored
        del payloads, payload, restore
        torch.cuda.empty_cache()


def prepare_chunks(engine, cfg, case, blob):
    p = cfg['adaptive']
    chunks = []
    ids = case['input_ids']
    for start in range(0, len(ids), p['chunk_size']):
        end = min(start + p['chunk_size'], len(ids))
        previous = blob[:, :, :, :start, :] if start else None
        text = engine.text_payload(ids[start:end])
        def text_work():
            decoded = engine.restore_text(text)
            return engine.prefill(decoded, previous)
        for _ in range(p['warmup']):
            text_work()
        text_times = []
        for _ in range(p['repeats']):
            restored, elapsed = timed(text_work)
            text_times.append(elapsed)
            del restored
        row = {'start': start, 'end': end, 'ids': ids[start:end], 'text_payload': text,
               'text': {'bytes': len(text), 'prefill_seconds': statistics.median(text_times),
                        'calibration_seconds': text_times, **engine.text_metadata(text)},
               'levels': {}, 'payloads': {}, 'restores': {}}
        chunk_blob = blob[:, :, :, start:end, :]
        for level in (1, 2, 3):
            payload, encode_s, restore = engine.encode(chunk_blob, f'cachegen{level}')
            for _ in range(p['warmup']):
                restore(payload)
            times = []
            for _ in range(p['repeats']):
                restored, elapsed = timed(lambda: restore(payload))
                validate_blob(restored, chunk_blob)
                times.append(elapsed)
                del restored
            row['levels'][level] = {'bytes': len(payload), 'decode_seconds': statistics.median(times),
                                     'calibration_seconds': times, 'offline_encode_seconds': encode_s}
            row['payloads'][level], row['restores'][level] = payload, restore
        chunks.append(row)
    return chunks


def execute_adaptive(engine, cfg, case, chunks, trace, strategy, slo, reserve):
    merged = None
    decisions = []
    estimated_transfer = 0.0
    torch.cuda.synchronize()
    start = time.perf_counter()
    for index, chunk in enumerate(chunks):
        elapsed = time.perf_counter() - start + estimated_transfer
        decision = next_decision(chunks, index, trace[index], slo, elapsed, strategy, reserve)
        level = decision['level']
        payload = chunk['payloads'][level] if level else chunk['text_payload']
        if cfg['adaptive']['mode'] == 'socket_serial':
            received, transfer_s = rate_limited_transfer(payload, trace[index])
        else:
            received = payload
            transfer_s, _ = representation_cost(chunk, level, trace[index])
            estimated_transfer += transfer_s
        def reconstruct():
            if level == 0:
                ids = engine.restore_text(received)
                return engine.prefill(ids, merged)
            decoded = chunk['restores'][level](received)
            return decoded if merged is None else torch.cat((merged, decoded), dim=3)
        merged, reconstruct_s = timed(reconstruct)
        decision.update(start=chunk['start'], end=chunk['end'], cache_tokens_after=merged.shape[3],
                        bytes=len(payload), transfer_seconds=transfer_s, reconstruct_seconds=reconstruct_s,
                        fallback_to_text=level == 0)
        if level == 0:
            decision.update(engine.text_metadata(payload))
        if merged.shape[3] != chunk['end']:
            raise RuntimeError('Adaptive cache position or tail was lost')
        decisions.append(decision)
    initial, first_s = timed(lambda: engine.first_state([], case['suffix_ids'], merged))
    wall = time.perf_counter() - start
    # Generation reuses the actual measured first token and resulting cache.
    generated = engine.generate(case['suffix_ids'], merged, cfg['adaptive']['max_new_tokens'], initial=initial)
    torch.cuda.synchronize()
    answer_wall = time.perf_counter() - start
    total = wall + estimated_transfer
    if not bool(torch.isfinite(merged).all()):
        raise RuntimeError('Adaptive merged cache contains nonfinite values')
    row = {**generated, **score_qa(generated['prediction'], case['answers']),
           'scored_answer': extract_answer(generated['prediction']), 'decisions': decisions,
           'levels': [d['level'] for d in decisions], 'text_fallback_chunks': sum(d['level'] == 0 for d in decisions),
           'mode': cfg['adaptive']['mode'], 'transfer_seconds': sum(d['transfer_seconds'] for d in decisions),
           'reconstruct_seconds': sum(d['reconstruct_seconds'] for d in decisions),
           'remaining_input_seconds': first_s, 'local_wall_seconds': wall, 'ttft_seconds': total,
           'answer_latency_seconds': answer_wall + estimated_transfer,
           'slo_seconds': slo, 'slo_violation': total > slo, 'first_token_reserve_seconds': reserve,
           'estimated_network_seconds': estimated_transfer,
           'historical_phrase_match': phrase_match(generated['prediction'], case['answer'])}
    if cfg['adaptive']['mode'] == 'socket_serial':
        row['measured_ttft_seconds'] = wall
    else:
        row['estimated_ttft_seconds'] = total
    return row


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
                key = f'{prefix}{trace_id}/{step["repetition"]}/{strategy}'
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
