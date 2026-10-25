"""Small, explicitly named mechanism task; independent of original baselines."""
import hashlib
import random
import statistics
import torch
from experiments import mechanisms
from experiments.config import ablation_cases
from experiments.engine import timed, validate_blob
from experiments.scoring import extract_answer, score_qa


def ablation(engine, cfg, selected, run, journal, job, progress):
    from experiments.tasks import case_fields, event
    p, task = cfg['ablation'], job['name']
    for case_index, (kind, case) in enumerate(ablation_cases(cfg, selected)):
        prefix = f'{task}/{kind}/{case["case_id"]}/'
        expected = len(p['precisions']) * len(p['entropy'])
        if sum(k.startswith(prefix) for k in journal.keys) == expected:
            continue
        context = case['context_ids'] if kind == 'quality' else case['input_ids']
        progress(task, prefix + 'offline-prefill')
        blob, prefill_s = timed(lambda: engine.prefill(context))
        for precision_index, precision in enumerate(p['precisions']):
            if all(f'{prefix}{precision}/{entropy}' in journal.keys for entropy in p['entropy']):
                continue
            q, quantize_s = timed(lambda: mechanisms.quantize(blob, engine.spec, precision))
            symbols_sha = hashlib.sha256(q.symbols.cpu().numpy().tobytes()).hexdigest()
            scales_sha = hashlib.sha256(q.scales.cpu().numpy().tobytes()).hexdigest()
            reference = mechanisms.dequantize(q.symbols, q.scales, q.bins, q.shape)
            order = list(p['entropy'])
            random.Random(cfg['seed'] + case_index * 100 + precision_index).shuffle(order)
            event(run, event='ablation_entropy_order', task=task, case_id=case['case_id'], precision=precision, order=order)
            for entropy in order:
                key = f'{prefix}{precision}/{entropy}'
                if key in journal.keys:
                    previous = next(r for r in journal.rows(task) if r['record_id'] == key)
                    if previous['symbols_sha256'] != symbols_sha or previous['scales_sha256'] != scales_sha:
                        raise ValueError('Resumed entropy pair has different quantized intermediates')
                    continue
                progress(task, key)
                for _ in range(p['warmup']):
                    payload, _ = mechanisms.serialize(q, entropy)
                    mechanisms.deserialize(payload)
                encode_times, decode_times = [], []
                for _ in range(p['repeats']):
                    (payload, metadata), encode_s = timed(lambda: mechanisms.serialize(q, entropy))
                    (symbols, restored), decode_s = timed(lambda: mechanisms.deserialize(payload))
                    encode_times.append(encode_s)
                    decode_times.append(decode_s)
                    # Full symbol equality includes every layer/channel/tail token.
                    if not torch.equal(symbols, q.symbols) or not torch.equal(restored, reference):
                        raise RuntimeError('Ablation entropy path changed symbols or reconstruction')
                validate_blob(restored, blob)
                row = {**case_fields(case), **metadata, 'kind': 'ablation', 'evaluation_kind': kind,
                    'method': f'ablation_{precision}_{entropy}', 'bytes': len(payload),
                    'offline_prefill_seconds': prefill_s, 'offline_prefill_scope': 'once_per_ablation_case_annotation',
                    'quantize_seconds': quantize_s, 'quantize_scope': 'once_per_case_precision_shared_by_entropy_paths',
                    'offline_encode_seconds': statistics.median(encode_times), 'decode_seconds': statistics.median(decode_times),
                    'encode_timings_seconds': encode_times, 'decode_timings_seconds': decode_times,
                    'timing_repeats': p['repeats'], 'symbols_sha256': symbols_sha, 'scales_sha256': scales_sha,
                    'symbols_identical': True, 'entropy_reconstructions_identical': True,
                    'raw_tensor_bytes': blob.numel() * blob.element_size(),
                    'kv_mae': float((blob - restored).float().abs().mean())}
                if kind == 'quality':
                    row.update(engine.nll(restored, context, case['continuation_ids']))
                else:
                    row.update(engine.generate(case['suffix_ids'], restored, cfg['quality']['max_new_tokens']))
                    row.update(score_qa(row['prediction'], case['answers']))
                    row['scored_answer'] = extract_answer(row['prediction'])
                journal.add(key, task, row)
                del symbols, restored, payload
            del q, reference
        del blob
        torch.cuda.empty_cache()
