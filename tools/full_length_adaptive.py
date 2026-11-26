"""Server-only full-context causal adaptation using real paced socket messages.

Sender encodes conversation KV only. Receiver reconstructs a mixed prefix and
computes the original question locally. No local execution or model downloads.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import time

from tools.full_length_qa import ROOT, load_rows, split_prompt, score


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def append(path, row):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()


def trace_for(template, chunks):
    if chunks < 1 or not template or any(not math.isfinite(v) or v <= 0 for v in template):
        raise ValueError('Invalid trace')
    if chunks == 1:
        return [template[0]]
    # Stretch the declared trajectory across the ACTUAL full context, including tail.
    return [template[(i * (len(template) - 1)) // (chunks - 1)] for i in range(chunks)]


def validate(cfg):
    if cfg.get('protocol') != 'full-length-adaptive-v1':
        raise ValueError('Wrong full-length protocol')
    if cfg['strategies'] != ['history', 'fixed1', 'fixed2', 'fixed3', 'text']:
        raise ValueError('Declare causal and all fixed precision/text controls')
    if cfg['quality_order'] != ['cachegen3', 'cachegen2', 'cachegen1']:
        raise ValueError('Keep author precision order, do not fit quality ranks on test answers')
    for key in ('chunk_tokens', 'generation_tokens', 'repeats', 'cost_repeats', 'probe_bytes', 'pacing_quantum_bytes'):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError('Invalid ' + key)
    if (not cfg['source_ids'] or len(set(cfg['source_ids'])) != len(cfg['source_ids'])
            or any(type(v) is not int or v < 0 for v in cfg['source_ids'])):
        raise ValueError('Invalid source IDs')
    if (not cfg['slo_seconds'] or len(set(cfg['slo_seconds'])) != len(cfg['slo_seconds'])
            or any(not math.isfinite(v) or v <= 0 for v in cfg['slo_seconds'])
            or not 0 <= cfg['safety_fraction'] < 1 or cfg['min_reserve_seconds'] < 0):
        raise ValueError('Invalid budgets')
    if not cfg['trace_templates_gbps']:
        raise ValueError('No traces')
    for values in cfg['trace_templates_gbps'].values():
        trace_for(values, 2)
    return cfg


def matrix(cfg, ids):
    return [(source, trace, slo, strategy, repeat)
            for source in ids for trace in cfg['trace_templates_gbps']
            for slo in cfg['slo_seconds'] for strategy in cfg['strategies']
            for repeat in range(cfg['repeats'])]


def predict_nll(engine, suffix, blob, targets):
    """Score raw-model answer tokens, conditioned on the fresh question, after TTFT."""
    import torch
    import torch.nn.functional as F
    if not targets:
        return None
    with torch.inference_mode():
        logits, cache = engine.forward(suffix + targets[:-1], blob, last_only=False)
        loss = F.cross_entropy(logits[0, -len(targets):].float(),
                               torch.tensor(targets, device='cuda'), reduction='mean')
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite answer NLL')
        result = float(loss)
        del logits, cache
        return result


def generate_native(engine, suffix, blob, limit, initial=None):
    """Greedy decoding without copying the entire KV tensor every token.

    Keep the model's native cache after the first question forward. The original
    context blob remains unchanged for the subsequent answer-NLL measurement.
    """
    import torch
    from transformers.cache_utils import DynamicCache
    generated = []
    reason = 'token_limit'
    with torch.inference_mode():
        token, cache = initial if initial is not None else engine.first_state([], suffix, blob)
        if not isinstance(cache, DynamicCache):
            cache = DynamicCache.from_legacy_cache(cache)
        for step in range(limit):
            if token == engine.tokenizer.eos_token_id:
                reason = 'eos'
                break
            generated.append(token)
            if step + 1 == limit:
                break
            previous = cache.get_seq_length()
            if previous + 1 > engine.spec['context_limit']:
                raise ValueError('Input/cache exceeds configured context window')
            positions = torch.arange(previous, previous + 1, device='cuda')
            out = engine.model.model(
                input_ids=torch.tensor([[token]], dtype=torch.long, device='cuda'),
                past_key_values=cache,
                attention_mask=torch.ones((1, previous + 1), dtype=torch.long, device='cuda'),
                position_ids=positions.unsqueeze(0), cache_position=positions,
                use_cache=True, return_dict=True)
            cache = out.past_key_values
            logits = engine.model.lm_head(out.last_hidden_state[:, -1:]).float()
            token = int(logits[0, -1].argmax())
            del logits, out
    return {'prediction': engine.tokenizer.decode(generated, skip_special_tokens=True,
                clean_up_tokenization_spaces=False), 'generated_ids': generated, 'stop_reason': reason}


def prepare_case(engine, context, host, cfg, folder):
    import torch
    from experiments.chunks import restore_chunk
    from experiments.engine import timed
    packets, costs = [], []
    rows = []
    for index, start in enumerate(range(0, len(context), cfg['chunk_tokens'])):
        end = min(start + cfg['chunk_tokens'], len(context))
        block = host[:, :, :, start:end, :].to('cuda')
        # Move only the current block to GPU; keep the full source KV on the host.
        item = {}
        for method in cfg['quality_order'] + ['text']:
            if method == 'text':
                payload, seconds = timed(lambda: engine.text_payload(context[start:end]))
                restore = engine.restore_text
            else:
                payload, seconds, restore = engine.encode(block, method)
            from experiments.chunks import PreparedChunk, frame
            meta = {'chunk_index': index, 'token_start': start, 'token_end': end, 'method': method,
                    'shape': list(block.shape), 'dtype': str(block.dtype), 'payload_bytes': len(payload)}
            wire = frame(payload, meta)
            packet = PreparedChunk(index, start, end, method, wire, restore, seconds, meta)
            item[method] = packet
            del payload, wire
        del block
        previous = host[:, :, :, :start, :].to('cuda') if start else None
        chunk_costs = {}
        for method, packet in item.items():
            # Always release the previous output before the next trial (24GB memory limit).
            restored, _ = timed(lambda: restore_chunk(engine, packet, packet.wire, previous))
            del restored
            measurements = []
            for _ in range(cfg['cost_repeats']):
                restored, seconds = timed(lambda: restore_chunk(engine, packet, packet.wire, previous))
                if restored.shape[3] != end:
                    raise ValueError('Preparation lost tokens')
                measurements.append(seconds)
                del restored
            chunk_costs[method] = {'bytes': len(packet.wire), 'restore_seconds': max(measurements)}
            rows.append({'chunk_index': index, 'start': start, 'end': end, 'method': method,
                         'message_sha256': packet.sha256, 'bytes': len(packet.wire),
                         'offline_encode_seconds': packet.encode_seconds,
                         'restore_runs_seconds': measurements})
        del previous
        packets.append(item)
        costs.append(chunk_costs)
        print('PREPARED_CHUNK', index, start, end, flush=True)
        torch.cuda.empty_cache()
    save(folder, {'chunks': rows, 'costs': costs,
                  'cost_assumption': 'offline known packet sizes; max of warm decode/prefix-prefill measurements on original prefixes'})
    return packets, costs


def replay(engine, cfg, packets, costs, trace, suffix, strategy, slo, first_reserve):
    import torch
    from experiments.chunks import restore_chunk
    from experiments.engine import timed
    from mini_repro.transport import transfer_chunks
    from paper_compression.policy import choose
    merged = None
    completed, decisions = [], []
    torch.cuda.synchronize()
    start = time.perf_counter()
    probe_data = bytes(cfg['probe_bytes'])
    probe_transfer = transfer_chunks([probe_data], trace[0], lambda _, data: None, mode='whole',
        origin=start, chunk_bytes=cfg['pacing_quantum_bytes'])
    if probe_transfer['received'] != [probe_data]:
        raise ValueError('Probe byte mismatch')
    probe = {'bytes': len(probe_data), 'seconds': probe_transfer['chunks'][0]['effective_transfer_seconds']}
    total_bytes = len(probe_data)
    for index, variants in enumerate(packets):
        decision = choose(costs, index, probe, completed, slo, time.perf_counter() - start,
                          first_reserve, cfg['safety_fraction'], cfg['min_reserve_seconds'], cfg['quality_order'])
        if strategy != 'history':
            decision['method'] = 'text' if strategy == 'text' else 'cachegen' + strategy[-1]
            decision['forecast_feasible'] = decision['candidate_forecasts_seconds'][decision['method']] <= decision['remaining_budget_seconds']
        packet = variants[decision['method']]
        def restore(_, wire):
            nonlocal merged
            merged = restore_chunk(engine, packet, wire, merged)
            torch.cuda.synchronize()
        # Environment/transport sees current bandwidth only AFTER the policy decided.
        received = transfer_chunks([packet.wire], trace[index], restore, mode='whole',
            origin=start, chunk_bytes=cfg['pacing_quantum_bytes'])
        if received['received'] != [packet.wire]:
            raise ValueError('Received packet differs')
        network = received['chunks'][0]
        completed.append({'bytes': len(packet.wire), 'seconds': network['effective_transfer_seconds']})
        total_bytes += len(packet.wire)
        decisions.append({**decision, 'method': packet.method, 'start': packet.start, 'end': packet.end,
                          'bytes': len(packet.wire), 'message_sha256': packet.sha256,
                          'configured_bandwidth_gbps': trace[index], 'network': network,
                          'byte_identity_checked': True})
    initial, first_seconds = timed(lambda: engine.first_state([], suffix, merged))
    wall = time.perf_counter() - start
    if merged.shape[3] != packets[-1][cfg['quality_order'][0]].end:
        raise ValueError('Reconstructed prefix is incomplete')
    return merged, initial, {'ttft_seconds': wall, 'slo_seconds': slo, 'slo_violation': wall > slo,
        'bytes_with_probe': total_bytes, 'probe': probe, 'decisions': decisions,
        'methods': [d['method'] for d in decisions], 'text_chunks': sum(d['method'] == 'text' for d in decisions),
        'infeasible_decisions': sum(not d['forecast_feasible'] for d in decisions),
        'first_token_seconds': first_seconds, 'first_token_reserve_seconds': first_reserve,
        'first_token_id': initial[0], 'mode': 'actual_paced_local_socket_serial_chunks'}


def summarize(output):
    with (output / 'records.jsonl').open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    groups = {}
    for row in rows:
        groups.setdefault((row['trace_id'], row['slo_seconds'], row['strategy']), []).append(row)
    result = []
    for key, group in sorted(groups.items()):
        method_counts = {}
        for row in group:
            for method in row['methods']:
                method_counts[method] = method_counts.get(method, 0) + 1
        nll = [r['delta_answer_nll'] for r in group if r['delta_answer_nll'] is not None]
        result.append({'trace_id': key[0], 'slo_seconds': key[1], 'strategy': key[2], 'runs': len(group),
            'independent_sources': len({r['source_id'] for r in group}),
            'mean_ttft_seconds': statistics.mean(r['ttft_seconds'] for r in group),
            'median_ttft_seconds': statistics.median(r['ttft_seconds'] for r in group),
            'slo_violation_rate': statistics.mean(r['slo_violation'] for r in group),
            'answer_accuracy': statistics.mean(r['includes_answer'] for r in group),
            'identical_to_raw_rate': statistics.mean(r['identical_to_raw'] for r in group),
            'mean_delta_answer_nll': statistics.mean(nll) if nll else None, 'chunk_method_counts': method_counts})
    save(output / 'summary.json', {'conditions': result, 'scope': 'full-context LongChat, real paced socket; not WAN',
         'repeats_are_independent_qa_samples': False, 'timing_includes_probe': True})


def run(args, cfg):
    import torch
    from experiments.engine import Engine, timed
    from paper_compression.provenance import provenance
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA server required')
    model_cfg = json.loads((ROOT / 'paper_compression/configs/core-study.json').read_text())
    torch.manual_seed(cfg['seed'])
    torch.set_num_threads(model_cfg['cpu_threads'])
    torch.backends.cuda.matmul.allow_tf32 = False
    # Tokenize ALL requested full contexts before reserving the run/model GPU memory.
    engine = Engine(model_cfg['model'], load=False, execution=model_cfg['execution'])
    rows = load_rows(args.ids)
    selected = {}
    for source in args.ids:
        context, suffix = split_prompt(engine.tokenizer, rows[source]['prompt'])['clean_question']
        if len(context) + len(suffix) + cfg['generation_tokens'] > model_cfg['model']['context_limit']:
            raise ValueError('Full context exceeds model window; do not truncate')
        selected[source] = {'context': context, 'suffix': suffix}
    args.output.mkdir(parents=True, exist_ok=False)
    source_hashes = provenance('stage2')
    for path in (Path(__file__), args.config, ROOT / 'tools/full_length_qa.py'):
        source_hashes[str(path.relative_to(ROOT))] = digest(path)
    data_hash = digest(ROOT / 'test_data/longchat.jsonl')
    meta = {'protocol': cfg['protocol'], 'config': cfg, 'source_ids': args.ids,
            'arm': 'clean_question', 'model': model_cfg['model'], 'execution': model_cfg['execution'],
            'source_hashes': source_hashes, 'dataset_sha256': data_hash,
            'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
            'started_utc': datetime.now(timezone.utc).isoformat(),
            'expected_records': len(matrix(cfg, args.ids)),
            'assumptions': ['offline known packet sizes and measured reconstruction costs',
              'probe traffic/time equalized across all five policies',
              'bandwidth trace changes per application chunk, not wall clock',
              'question stays on receiver; only context KV/text crosses socket',
              'paper SLOs .5/1s plus preregistered full-context diagnostic budgets 2/4s']}
    save(args.output / 'run_meta.json', meta)
    save(args.output / 'selection.json', selected)
    start = time.monotonic()
    count = 0
    def status(state, **extra):
        save(args.output / 'status.json', {'status': state, 'completed_records': count,
             'expected_records': meta['expected_records'], 'wall_seconds': time.monotonic() - start, **extra})
    status('running')
    try:
        engine.load()
        for source in args.ids:
            print('SOURCE', source, 'FULL_TOKENS', len(selected[source]['context']), flush=True)
            context, suffix = selected[source]['context'], selected[source]['suffix']
            raw_blob = engine.prefill(context)
            baseline = generate_native(engine, suffix, raw_blob, cfg['generation_tokens'])
            baseline_nll = predict_nll(engine, suffix, raw_blob, baseline['generated_ids'])
            first_runs = []
            for _ in range(cfg['cost_repeats']):
                _, seconds = timed(lambda: engine.first(context, suffix, raw_blob))
                first_runs.append(seconds)
            host = raw_blob.cpu()
            del raw_blob
            torch.cuda.empty_cache()
            packets, costs = prepare_case(engine, context, host, cfg, args.output / f'preparation-{source}.json')
            del host
            append(args.output / 'raw.jsonl', {'source_id': source, 'kv_tokens': len(context),
                   **baseline, **score(baseline['prediction'], rows[source]['label']),
                   'answer_nll': baseline_nll, 'first_token_runs': first_runs})
            traces = {name: trace_for(values, len(packets)) for name, values in cfg['trace_templates_gbps'].items()}
            schedule = matrix(cfg, [source])
            random.Random(cfg['seed'] + source).shuffle(schedule)
            save(args.output / f'schedule-{source}.json', {'schedule': schedule, 'traces_gbps': traces})
            for _, trace_id, slo, strategy, repeat in schedule:
                merged, initial, timing = replay(engine, cfg, packets, costs, traces[trace_id], suffix,
                                                 strategy, slo, max(first_runs))
                generated = generate_native(engine, suffix, merged, cfg['generation_tokens'], initial=initial)
                del initial
                nll = predict_nll(engine, suffix, merged, baseline['generated_ids'])
                row = {'source_id': source, 'kv_tokens': len(context), 'suffix_tokens': len(suffix),
                       'trace_id': trace_id, 'strategy': strategy, 'repetition': repeat,
                       **timing, **generated, **score(generated['prediction'], rows[source]['label']),
                       'identical_to_raw': generated['prediction'] == baseline['prediction'],
                       'answer_nll': nll, 'delta_answer_nll': nll - baseline_nll if nll is not None else None,
                       'quality_evaluation_outside_ttft': True}
                append(args.output / 'records.jsonl', row)
                count += 1
                status('running', source_id=source)
                print('RESULT', count, '/', meta['expected_records'], source, trace_id, slo, strategy,
                      f'{timing["ttft_seconds"]:.3f}s', 'CORRECT', int(row['includes_answer']),
                      'CHOICES', ','.join(timing['methods']), flush=True)
                del merged
                torch.cuda.empty_cache()
            del packets, costs
            torch.cuda.empty_cache()
        current = provenance('stage2')
        current.update({str(path.relative_to(ROOT)): digest(path)
                        for path in (Path(__file__), args.config, ROOT / 'tools/full_length_qa.py')})
        if current != source_hashes or digest(ROOT / 'test_data/longchat.jsonl') != data_hash:
            raise ValueError('Sources/data changed during run')
        with (args.output / 'records.jsonl').open() as stream:
            result = [json.loads(line) for line in stream if line.strip()]
        keys = {(r['source_id'], r['trace_id'], r['slo_seconds'], r['strategy'], r['repetition']) for r in result}
        if len(result) != len(keys) or keys != set(matrix(cfg, args.ids)):
            raise ValueError('Missing/duplicate experiment conditions')
        summarize(args.output)
        status('completed')
    except BaseException as exc:
        status('failed', error_type=type(exc).__name__, error=str(exc))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'paper_compression/configs/full-length-adaptive.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ids', type=int, nargs='+')
    args = parser.parse_args()
    cfg = validate(json.loads(args.config.read_text()))
    args.ids = args.ids if args.ids is not None else cfg['source_ids']
    if not args.ids or len(set(args.ids)) != len(args.ids) or not set(args.ids) <= set(cfg['source_ids']):
        parser.error('Use distinct preregistered source IDs')
    args.config = args.config.resolve()
    run(args, cfg)


if __name__ == '__main__':
    main()
