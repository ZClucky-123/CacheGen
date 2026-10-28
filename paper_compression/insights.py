"""§5.1 diagnostics, using validation documents only; no test-set tuning."""
import torch
from .config import CodecConfig
from .probability import grouped_histogram, make_cdf
from .quantization import quantize, delta_symbols


def distribution(values, stride):
    rows = values.shape[0]
    flat = values.reshape(rows, -1)
    samples = flat[:, ::stride].abs().reshape(-1)
    return {'mean_abs': float(flat.abs().mean()),
            'variance_by_kv_layer': flat.var(dim=1, unbiased=False).tolist(),
            'abs_quantiles': torch.quantile(samples, torch.tensor([.5, .9, .99], device=values.device)).tolist()}


def conditional_entropy(counts):
    counts = counts.double().reshape(-1, counts.shape[-1])
    totals = counts.sum(-1)
    numerator = (totals * totals.clamp_min(1).log2()).sum() - (counts * counts.clamp_min(1).log2()).sum()
    return float(numerator / totals.sum())


def cross_entropy(counts, cdf):
    """Held-out counts scored using a frozen calibration CDF (no refitting)."""
    boundaries = cdf.to(device=counts.device, dtype=torch.int32).bitwise_and(65535)
    boundaries[..., -1] = 65536
    if tuple(boundaries.shape[:-1]) != tuple(counts.shape[:-1]):
        raise ValueError('Calibration and held-out grouping geometries differ')
    frequencies = boundaries[..., 1:] - boundaries[..., :-1]
    if not bool((frequencies > 0).all()) or int(counts.sum()) == 0:
        raise ValueError('Invalid grouping CDF or empty validation counts')
    cost = 16. - frequencies.double().log2()
    return float((counts.double() * cost).sum() / counts.sum())


def run(args, cfg):
    from tools.paper_study import run_folder, inputs, qa_cases, score_answer
    from tools.paper_compression import setup, save, append
    from .config import fingerprint
    p = cfg['insights']
    # Matched uniform quantizer prevents layer-specific alphabets confounding grouping.
    fixed = CodecConfig(layers=cfg['model']['layers'], vector_range=p['vector_range'],
                        delta_bin_widths=(1., 1., 1.), use_delta=False)
    histograms = {}
    with run_folder(args, cfg) as data_hash:
        engine, identity = setup(cfg)
        selected = inputs(engine, cfg, 'validation', [p['tokens']])
        qa = qa_cases(engine, cfg, 'validation')
        save(args.output / 'model_identity.json', identity)
        save(args.output / 'selection.json', selected)
        engine.load()
        calibration = inputs(engine, cfg, 'calibration', [p['tokens']])
        save(args.output / 'grouping-calibration-selection.json', calibration)
        for case in calibration:
            print('GROUPING CALIBRATION', case['source_id'], flush=True)
            blob = engine.prefill(case['context_ids'])
            q = quantize(blob, fixed)
            symbols, _ = delta_symbols(q.delta, fixed)
            for grouping in ('global', 'layer', 'layer_channel', 'token'):
                counts = grouped_histogram(symbols, grouping)
                if grouping not in histograms:
                    histograms[grouping] = counts
                else:
                    histograms[grouping] += counts
            del blob, q, symbols, counts
            torch.cuda.empty_cache()
        cdfs = {g: make_cdf(c.cpu()) for g, c in histograms.items()}
        histograms.clear()
        torch.save({'format': 'paper-grouping-held-out-v1', 'cdfs': cdfs,
                    'codec': fixed.to_dict(), 'model_identity': identity, 'dataset_sha256': data_hash,
                    'calibration_source_ids': cfg['splits']['calibration'],
                    'calibration_token_hashes': {str(c['source_id']): fingerprint(c['context_ids']) for c in calibration}},
                   args.output / 'grouping-cdfs.pt')
        for case in selected:
            source, context, continuation = case['source_id'], case['context_ids'], case['continuation_ids']
            print('INSIGHTS', source, flush=True)
            blob = engine.prefill(context)
            raw = engine.nll(blob, context, continuation)
            raw_qa = score_answer(engine, blob, qa[source])
            groups = blob.permute(1, 0, 3, 2, 4).reshape(2 * fixed.layers, len(context), -1).float()
            for lag in (0, 1, 9):
                values = groups if lag == 0 else groups[:, lag:] - groups[:, :-lag]
                append(args.output / 'locality.jsonl', {'source_id': source, 'lag': lag,
                       'representation': 'raw' if lag == 0 else 'pairwise_delta',
                       **distribution(values, p['sample_stride'])})
                if lag:
                    del values
            pos = torch.arange(len(context), device=blob.device)
            nonanchor = pos.remainder(10) != 0
            residual = groups[:, nonanchor] - groups[:, (pos[nonanchor] // 10) * 10]
            append(args.output / 'locality.jsonl', {'source_id': source, 'representation': 'anchor_delta',
                   **distribution(residual, p['sample_stride'])})
            del residual, groups
            boundaries = (0,) + fixed.boundaries
            for group in range(3):
                start, end = boundaries[group:group + 2]
                # Same normalized perturbation at one layer group only, K and V together.
                altered = blob.clone()
                block = altered[start:end].float()
                vectors = block.permute(0, 1, 3, 2, 4).reshape(end - start, 2, len(context), -1)
                maxima = vectors.abs().amax(-1, keepdim=True).clamp_min(1e-12)
                restored = torch.round(vectors / maxima * p['vector_range']) / p['vector_range'] * maxima
                normalized_mse = float(((restored - vectors) / maxima).square().mean())
                altered[start:end] = restored.reshape(end - start, 2, len(context),
                    blob.shape[2], blob.shape[4]).permute(0, 1, 3, 2, 4).half()
                score = engine.nll(altered, context, continuation)
                append(args.output / 'sensitivity.jsonl', {'source_id': source, 'layer_group': group,
                       'layer_start': start, 'layer_end': end, 'raw_nll': raw['nll'],
                       'delta_nll': score['nll'] - raw['nll'], 'normalized_mse': normalized_mse,
                       'raw_qa': raw_qa, 'perturbed_qa': score_answer(engine, altered, qa[source]),
                       'perturbation': 'same normalized rounding applied to one group only'})
                del altered, block, vectors, maxima, restored
            q = quantize(blob, fixed)
            symbols, _ = delta_symbols(q.delta, fixed)
            for grouping in ('global', 'layer', 'layer_channel', 'token'):
                counts = grouped_histogram(symbols, grouping)
                if grouping not in histograms:
                    histograms[grouping] = counts
                else:
                    histograms[grouping] += counts
            del blob, q, symbols
            torch.cuda.empty_cache()
        save(args.output / 'grouping.json', {'split': 'validation', 'codec': fixed.to_dict(),
             'calibration_source_ids': cfg['splits']['calibration'],
             'validation_source_ids': cfg['splits']['validation'],
             'metric': 'held-out cross entropy using frozen 16bit calibration CDFs',
             'includes_cdf_storage_cost': False, 'is_held_out_coding_size': True,
             'size_kind': 'theoretical_symbol_bits_not_actual_arithmetic_bytes',
             'cdf_tensor_bytes': {g: c.numel() * c.element_size() for g, c in cdfs.items()},
             'bits_per_symbol': {g: cross_entropy(c, cdfs[g]) for g, c in histograms.items()}})
        save(args.output / 'interpretation.json', {'scope': 'Mistral FP16, validation documents, reduced scale',
             'locality': 'Compare within each KV layer; report every source, do not assume all layers improve',
             'sensitivity': 'Compare group delta NLL, normalized MSE and first-topic EM/F1; reduced-scale validation workload',
             'grouping': 'Calibrate CDFs on IDs30..49; score IDs20..29 without fitting validation distributions',
             'parameter_status': cfg['parameter_status']})
