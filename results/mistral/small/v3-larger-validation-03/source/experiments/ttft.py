"""Paired whole / chunk-serial / bounded chunk-pipeline TTFT experiment."""
import hashlib
import json
import os
import shutil
import time
import torch
from experiments.chunks import PreparedChunk, prepare, preparation_metadata, restore_chunk, unframe
from experiments.config import ttft_variants
from experiments.engine import timed, validate_blob
from experiments.records import save, ttft_key
from mini_repro.transport import transfer_chunks, overlap_timeline
from mini_repro.ttft import measurement_schedule
from experiments.numerics import cached_reference, audit_generation, compare_kv


def replay(engine, cfg, packets, bandwidth, mode, context, suffix):
    merged = None
    whole_text = mode == 'whole' and packets[0].method == 'text'
    torch.cuda.synchronize()
    start = time.perf_counter()  # Before connection creation and CPU threads.
    def restore(index, wire):
        nonlocal merged
        if whole_text:
            # Preserve the original whole-message reference: retokenize, then
            # ONE optimized context+suffix forward. Chunk controls both execute
            # dependent prefill per chunk, followed by suffix-only inference.
            metadata, payload = unframe(wire)
            if metadata != packets[index].metadata:
                raise ValueError('Whole text metadata differs')
            merged = engine.restore_text(payload)
        else:
            merged = restore_chunk(engine, packets[index], wire, merged)
        torch.cuda.synchronize()  # Includes completed GPU work, not submission.
        return merged
    result = transfer_chunks([c.wire for c in packets], bandwidth, restore, mode=mode,
        queue_capacity=cfg['transport']['queue_capacity'],
        chunk_bytes=cfg['transport']['pacing_quantum_bytes'], origin=start, analyze_timeline=False)
    token, first_s = timed(lambda: engine.first(merged, suffix) if whole_text else engine.first(context, suffix, merged))
    wall = time.perf_counter() - start  # CPU integer token is ready.
    # All equality, finite checks and metadata enrichment are outside TTFT.
    result.update(overlap_timeline(result['chunks']))
    if result['received'] != [c.wire for c in packets]:
        raise RuntimeError('TTFT socket bytes differ')
    if whole_text and merged != context:
        raise RuntimeError('Whole text token identity differs')
    for chunk, packet in zip(result['chunks'], packets):
        chunk.update(packet.metadata)
    transfer_s = sum(c['effective_transfer_seconds'] for c in result['chunks'])
    row = {**preparation_metadata(packets), 'mode': mode, 'chunks': result['chunks'],
        'transfer_seconds': transfer_s, 'reconstruct_seconds': sum(c['restore_seconds'] for c in result['chunks']),
        'main_queue_wait_seconds': sum(c['main_queue_wait_seconds'] for c in result['chunks']),
        'receiver_capacity_wait_seconds': sum(c['capacity_wait_seconds'] for c in result['chunks']),
        'send_permission_wait_seconds': sum(c['send_permission_wait_seconds'] for c in result['chunks']),
        'send_blocking_seconds': sum(c['send_blocking_seconds'] for c in result['chunks']),
        'pacing_wait_seconds': sum(c['pacing_wait_seconds'] for c in result['chunks']),
        'overlap_seconds': result['overlap_seconds'], 'overlap_intervals': result['overlap_intervals'],
        'queue_capacity': result['queue_capacity'], 'pacing_quantum_bytes': result['pacing_quantum_bytes'],
        'remaining_input_seconds': first_s, 'ttft_wall_seconds': wall,
        'actual_bandwidth_gbps': sum(len(c.wire) for c in packets) * 8 / max(transfer_s, 1e-12) / 1e9,
        'first_token_id': token, 'byte_identity_checked': True}
    row['actual_to_target'] = row['actual_bandwidth_gbps'] / bandwidth if bandwidth else None
    return merged, row


def _persist(folder, packets, prefill_s):
    folder.mkdir(parents=True, exist_ok=True)
    entries = []
    for c in packets:
        name = f'{c.index}.bin'
        with (folder / name).open('wb') as stream:
            stream.write(c.wire)
            stream.flush()
            os.fsync(stream.fileno())
        entries.append({'index': c.index, 'start': c.start, 'end': c.end, 'method': c.method,
                        'encode_seconds': c.encode_seconds, 'metadata': c.metadata, 'file': name, 'sha256': c.sha256})
    save(folder / 'prepared.json', {'packets': entries, 'offline_prefill_seconds': prefill_s})


def _load(engine, folder):
    obj = json.loads((folder / 'prepared.json').read_text())
    packets = []
    for c in obj['packets']:
        wire = (folder / c['file']).read_bytes()
        if hashlib.sha256(wire).hexdigest() != c['sha256']:
            raise ValueError('Persisted prepared TTFT bytes corrupted')
        packets.append(PreparedChunk(c['index'], c['start'], c['end'], c['method'], wire,
            engine.restore_function(c['method'], c['end'] - c['start']), c['encode_seconds'], c['metadata']))
    return tuple(packets), obj['offline_prefill_seconds']


def ttft(engine, cfg, selected, run, journal, job, progress):
    from experiments.tasks import event, case_fields
    task, p = job['name'], cfg['ttft']
    checks_path = run / 'ttft_checks.json'
    checks = json.loads(checks_path.read_text()) if checks_path.exists() else []
    for index, case in enumerate(selected['ttft']):
        prefix = f'{task}/{case["case_id"]}/'
        variants = list(ttft_variants(cfg, len(case['context_ids'])))
        expected = len(p['methods']) * len(p['bandwidth_gbps']) * p['repeats'] * len(variants)
        pending = run / 'pending-prepared' / 'ttft' / case['case_id']
        if sum(k.startswith(prefix) for k in journal.keys) == expected:
            if pending.exists():
                shutil.rmtree(pending)
            continue
        context, suffix = case['context_ids'], case['suffix_ids']
        progress(task, f'{prefix}offline-prefill')
        blob, prefill_s = timed(lambda: engine.prefill(context))
        direct = engine.first(context, suffix)
        if engine.first(context, suffix, blob) != direct:
            raise RuntimeError('TTFT raw cache parity failed')
        prepared = {}
        # Whole is an independent reference. Both chunk modes reuse each tuple.
        for size in sorted({size for _, size in variants}):
            for method in p['methods']:
                folder = pending / str(size) / method
                if (folder / 'prepared.json').exists():
                    packets, original_prefill_s = _load(engine, folder)
                else:
                    packets = prepare(engine, context, blob, method, size)
                    original_prefill_s = prefill_s
                    _persist(folder, packets, original_prefill_s)
                prepared[(size, method)] = (packets, original_prefill_s)
                # Check the actual threaded transport, tails, restored tensors,
                # and multistep output against the SAME encoded representation.
                serial, _ = replay(engine, cfg, packets, None, 'chunk_serial', context, suffix)
                pipelined, _ = replay(engine, cfg, packets, None, 'chunk_pipeline', context, suffix)
                validate_blob(serial, blob)
                if not torch.equal(serial, pipelined):
                    raise RuntimeError('Serial/pipeline restored KV differs for same bytes')
                limit = cfg['checks']['generation_tokens']
                generated = engine.generate(suffix, serial, limit)['generated_ids']
                if generated != engine.generate(suffix, pipelined, limit)['generated_ids']:
                    raise RuntimeError('Serial/pipeline generation differs')
                text_error = None
                text_generation_audit = None
                text_kv_audit = None
                if method == 'raw' and not torch.equal(serial, blob):
                    raise RuntimeError('Raw chunk assembly differs')
                if method == 'text':
                    text_error = float((serial - blob).float().abs().max())
                    text_kv_audit = compare_kv(blob, serial)
                    if not text_kv_audit['elementwise_allclose']:
                        raise RuntimeError('Sequential text prefill exceeds numerical bounds')
                    reference_ids, reference_scores = cached_reference(engine, suffix, blob, limit)
                    text_generation_audit = audit_generation(engine, suffix, serial, limit,
                                                             reference_ids, reference_scores, generated)
                    del reference_scores
                checks.append({'case_id': case['case_id'], 'chunk_size': size, 'method': method,
                    **preparation_metadata(packets), 'byte_identity': True, 'mode_kv_identity': True,
                    'mode_generated_ids': generated, 'text_vs_whole_prefill_max_abs': text_error,
                    'text_vs_whole_generation_audit': text_generation_audit,
                    'text_vs_whole_kv_audit': text_kv_audit,
                    'raw_reassembly_identical': method == 'raw', 'cache_tokens': serial.shape[3]})
                save(checks_path, checks)
                del serial, pipelined
        del blob
        labels = [f'{method}:{mode}:{size}' for method in p['methods'] for mode, size in variants]
        schedule = measurement_schedule(labels, p['bandwidth_gbps'], warmup=p['warmup'],
            repeats=p['repeats'], seed=cfg['seed'] + index)
        event(run, event='ttft_schedule', task=task, case_id=case['case_id'], schedule=schedule)
        for step_index, step in enumerate(schedule):
            method, mode, size = step['method'].split(':')
            size = int(size)
            key = ttft_key(task, case['case_id'], step['bandwidth_gbps'], step['repetition'], method, mode, size,
                           cfg['transport']['queue_capacity'])
            if step['phase'] == 'measure' and key in journal.keys:
                continue
            progress(task, key)
            packets, original_prefill_s = prepared[(size, method)]
            merged, row = replay(engine, cfg, packets, step['bandwidth_gbps'], mode, context, suffix)
            row.update(case_fields(case), kind='ttft', **{k: v for k, v in step.items() if k != 'method'},
                method=method, chunk_size=size, schedule_position=step_index,
                offline_prefill_seconds=original_prefill_s, offline_prefill_scope='once_per_context_annotation',
                offline_encode_scope='once_per_method_chunk_geometry_shared_by_modes', matches_raw=row['first_token_id'] == direct)
            if method == 'text':
                from experiments.chunks import unframe
                metas = [engine.text_metadata(unframe(c.wire)[1]) for c in packets]
                row.update(text_wire='utf8_token_patch_v1', token_patch_count=sum(m['token_patch_count'] for m in metas),
                           patch_inserted_tokens=sum(m['patch_inserted_tokens'] for m in metas))
            if step['phase'] == 'measure':
                journal.add(key, task, row)
            else:
                event(run, event='warmup', task=task, observation=row)
            del merged
        # Incomplete pairs keep their exact bytes for crash-safe resume. Only
        # after every formal row is durable may this bounded temporary cache go.
        shutil.rmtree(pending)
        del prepared
        torch.cuda.empty_cache()
