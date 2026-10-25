"""Strict, model-neutral summaries; clustered intervals preserve shared sources."""
import csv
import json
import random
import statistics as stats
import sys
from collections import defaultdict
from pathlib import Path
from experiments.records import save


def quantile(values, q):
    values = sorted(values)
    rank = (len(values) - 1) * q
    lo = int(rank)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (rank - lo)


def cluster_interval(pairs, seed=0, draws=2000):
    groups = defaultdict(list)
    for source, value in pairs:
        groups[str(source)].append(value)
    keys = sorted(groups)
    if len(keys) < 2:
        return None
    rng = random.Random(seed)
    values = [stats.mean(v for key in rng.choices(keys, k=len(keys)) for v in groups[key]) for _ in range(draws)]
    return [quantile(values, .025), quantile(values, .975)]


def csv_file(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ['empty'])
        writer.writeheader()
        writer.writerows(rows)


def summarize(cfg, rows):
    quality = [r for r in rows if r['kind'] == 'quality']
    controlled = [r for r in rows if r['kind'] == 'controlled']
    raw = {r['case_id']: r for r in quality if r['method'] == 'raw'}
    raw_qa = {r['case_id']: r for r in controlled if r['method'] == 'raw'}
    qrows = []
    for length in cfg['quality']['lengths']:
        for method in cfg['quality']['methods']:
            selected = [r for r in quality if r['context_tokens'] == length and r['method'] == method]
            qa = [r for r in controlled if r['context_tokens'] == length and r['method'] == method]
            correct = [r for r in qa if raw_qa[r['case_id']]['em'] == 1]
            retention = stats.mean(r['em'] for r in correct) if correct else None
            delta = stats.mean(r['nll'] - raw[r['case_id']]['nll'] for r in selected)
            gate = 'unassessable' if retention is None else 'pass' if retention >= .9 and delta <= .02 else 'fail'
            pairs = [(r['source_id'], r['nll'] - raw[r['case_id']]['nll']) for r in selected]
            qrows.append({'context_tokens': length, 'method': method, 'independent_source_documents': len(set(r['source_id'] for r in selected)),
                'natural_conditions': len(selected), 'mean_bytes': stats.mean(r['bytes'] for r in selected),
                'raw_serialized_over_method': sum(raw[r['case_id']]['bytes'] for r in selected) / sum(r['bytes'] for r in selected),
                'raw_tensor_over_method': sum(r['raw_tensor_bytes'] for r in selected) / sum(r['bytes'] for r in selected),
                'mean_nll': stats.mean(r['nll'] for r in selected), 'paired_nll_delta': delta,
                'paired_nll_cluster_ci95': cluster_interval(pairs, cfg['seed']),
                'controlled_fact_sources': len(set(r['source_id'] for r in qa)), 'controlled_position_conditions': len(qa),
                'raw_em_correct': len(correct), 'em_retention': retention, 'acceptance': gate,
                'mean_em': stats.mean(r['em'] for r in qa), 'mean_f1': stats.mean(r['f1'] for r in qa),
                'mean_offline_encode_seconds': stats.mean(r['offline_encode_seconds'] for r in selected),
                'mean_offline_prefill_seconds': stats.mean(r['offline_prefill_seconds'] for r in selected)
                                               if all('offline_prefill_seconds' in r for r in selected) else None,
                'mean_diagnostic_decode_seconds': stats.mean(r['decode_seconds'] for r in selected)})
    qa_rows = []
    for kind in ('controlled', 'triviaqa'):
        q = [r for r in rows if r['kind'] == kind]
        baseline = {r['case_id']: r for r in q if r['method'] == 'raw'}
        for method in cfg['quality']['methods']:
            selected = [r for r in q if r['method'] == method]
            deltas = [(r['source_id'], r['em'] - baseline[r['case_id']]['em']) for r in selected]
            qa_rows.append({'dataset': kind, 'method': method, 'independent_questions_or_facts': len(set(r['source_id'] for r in selected)),
                            'conditions': len(selected), 'mean_em': stats.mean(r['em'] for r in selected),
                            'mean_f1': stats.mean(r['f1'] for r in selected),
                            'paired_em_delta': stats.mean(d for _, d in deltas),
                            'paired_em_cluster_ci95': cluster_interval(deltas, cfg['seed'])})
    ttft = defaultdict(list)
    for row in rows:
        if row['kind'] == 'ttft':
            ttft[(row['context_tokens'], row['bandwidth_gbps'], row['method'], row['mode'], row['chunk_size'], row['queue_capacity'])].append(row)
    medians = {key: stats.median(r['ttft_wall_seconds'] for r in group) for key, group in ttft.items()}
    timing_rows = []
    for (length, bw, method, mode, chunk_size, capacity), group in sorted(ttft.items()):
        values = [r['ttft_wall_seconds'] for r in group]
        raw_method = 'raw' if method == 'text' else method
        gate = next(r['acceptance'] for r in qrows if r['context_tokens'] == length and r['method'] == raw_method) if length in cfg['quality']['lengths'] else 'unassessable'
        timing_rows.append({'context_tokens': length, 'target_bandwidth_gbps': bw, 'method': method, 'mode': mode, 'chunk_size': chunk_size, 'queue_capacity': capacity,
            'independent_source_documents': len(set(r['source_id'] for r in group)), 'timing_observations': len(group),
            'repeats_per_condition': cfg['ttft']['repeats'], 'median_seconds': stats.median(values), 'p95_descriptive_seconds': quantile(values, .95),
            'median_cluster_ci95': _median_ci(group, cfg['seed']),
            'mean_transfer_seconds': stats.mean(r['transfer_seconds'] for r in group),
            'mean_reconstruct_seconds': stats.mean(r['reconstruct_seconds'] for r in group),
            'mean_remaining_input_seconds': stats.mean(r['remaining_input_seconds'] for r in group),
            'mean_offline_prefill_once_per_context_seconds': stats.mean(r['offline_prefill_seconds'] for r in group),
            'mean_offline_encode_once_per_context_seconds': stats.mean(r['offline_encode_seconds'] for r in group),
            'mean_actual_bandwidth_gbps': stats.mean(r['actual_bandwidth_gbps'] for r in group),
            'actual_target_ratio': stats.mean(r['actual_to_target'] for r in group),
            'speedup_vs_text': medians[(length, bw, 'text', mode, chunk_size, capacity)] / stats.median(values),
            'speedup_vs_uniform8': medians[(length, bw, 'uniform8', mode, chunk_size, capacity)] / stats.median(values),
            'same_length_quality_acceptance': gate})
    adaptive = defaultdict(list)
    for row in rows:
        if row['kind'] == 'adaptive':
            adaptive[(row['task'], row['strategy'], row['throughput_estimator'], row['chunk_size'])].append(row)
    arows = []
    for (task, strategy, estimator, chunk_size), group in sorted(adaptive.items()):
        conditions = defaultdict(list)
        for r in group:
            conditions[(r['case_id'], r['trace_id'])].append(r)
        raw_correct_conditions = [items for items in conditions.values() if items[0]['raw_em'] == 1]
        arows.append({'task': task, 'strategy': strategy, 'mode': cfg['adaptive']['mode'], 'throughput_estimator': estimator, 'chunk_size': chunk_size,
            'independent_facts': len(set(r['source_id'] for r in group)), 'fact_trace_conditions': len(conditions),
            'timing_observations': len(group), 'repeats_per_condition': cfg['adaptive']['repeats'],
            'median_ttft_seconds': stats.median(r['ttft_seconds'] for r in group),
            'p95_descriptive_seconds': quantile([r['ttft_seconds'] for r in group], .95),
            'slo_violation_rate': stats.mean(r['slo_violation'] for r in group),
            'slo_rate_fact_cluster_ci95': cluster_interval([(r['source_id'], float(r['slo_violation'])) for r in group], cfg['seed']),
            'condition_mean_em': stats.mean(stats.mean(r['em'] for r in items) for items in conditions.values()),
            'condition_mean_f1': stats.mean(stats.mean(r['f1'] for r in items) for items in conditions.values()),
            'raw_condition_mean_em': stats.mean(items[0]['raw_em'] for items in conditions.values()),
            'raw_correct_fact_trace_conditions': len(raw_correct_conditions),
            'raw_em_retention': stats.mean(stats.mean(r['em'] for r in items) for items in raw_correct_conditions)
                                if raw_correct_conditions else None,
            'paired_em_delta': stats.mean(r['em'] - r['raw_em'] for r in group),
            'paired_em_fact_cluster_ci95': cluster_interval([(r['source_id'], r['em'] - r['raw_em']) for r in group], cfg['seed']),
            'text_fallback_chunks': sum(r['text_fallback_chunks'] for r in group),
            'median_answer_latency_seconds': stats.median(r['answer_latency_seconds'] for r in group)})
    # Resample source documents jointly across lengths to preserve their correlation.
    across = {m: cluster_interval([(r['source_id'], r['nll'] - raw[r['case_id']]['nll']) for r in quality if r['method'] == m], cfg['seed'])
              for m in cfg['quality']['methods']}
    from experiments.analysis_v3 import pipeline_pairs, pipeline_summary, ablation_summary
    paired = pipeline_pairs(rows)
    ablation, entropy_pairs = ablation_summary(rows)
    return {'ttft_pairs': paired, 'pipeline_comparison': pipeline_summary(paired, cfg['seed']),
            'ablation': ablation, 'entropy_pairs': entropy_pairs, 'protocol': cfg['protocol'], 'quality': qrows, 'qa': qa_rows, 'ttft': timing_rows,
            'adaptive': arows, 'all_lengths_paired_nll_source_cluster_ci95': across,
            'interval_note': '2000 seeded source-cluster bootstrap draws; with few sources these are descriptive, not equivalence evidence'}


def _median_ci(group, seed):
    groups = defaultdict(list)
    for r in group:
        groups[r['source_id']].append(r['ttft_wall_seconds'])
    keys = sorted(groups)
    if len(keys) < 2:
        return None
    rng = random.Random(seed)
    values = [stats.median(v for key in rng.choices(keys, k=len(keys)) for v in groups[key]) for _ in range(2000)]
    return [quantile(values, .025), quantile(values, .975)]


def table(rows, columns):
    def fmt(v):
        return '—' if v is None else f'{v:.5g}' if isinstance(v, float) else str(v).replace('|', '\\|')
    return '\n'.join(['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join('---' for _ in columns) + ' |'] +
                     ['| ' + ' | '.join(fmt(r[c]) for c in columns) + ' |' for r in rows])


def report(run, cfg, journal, expected):
    journal.validate([key for keys in expected.values() for key in keys])
    result = summarize(cfg, journal.rows())
    save(run / 'summary.json', result)
    for name in ('quality', 'qa', 'ttft', 'adaptive', 'ttft_pairs', 'pipeline_comparison', 'ablation', 'entropy_pairs'):
        csv_file(run / f'{name}.csv', result[name])
    from experiments.analysis_v3 import timeline_rows, extra_plots
    timeline, overlap = timeline_rows(journal.rows())
    csv_file(run / 'ttft_timeline.csv', timeline)
    csv_file(run / 'ttft_overlap.csv', overlap)
    text = [f'# {cfg["model"]["name"]} / {cfg["scale"]} / {cfg["protocol"]}',
        f'Model: {cfg["model"]["id"]} @ {cfg["model"]["revision"]}; weights: {cfg["model"]["weight_loading"]}.',
        f'All {len(expected)} core tasks and {sum(map(len, expected.values()))} expected observations verified. TriviaQA is supplementary inside the quality task.',
        '## Protocol and limits',
        'NLL scores every held-out token 1..N. The final context token is replayed against restored KV[0:L-1]; no historical v1 NLL is pooled. Negative ΔNLL alone does not show improved model quality.',
        'QA is deterministic greedy decoding until EOS or the configured token limit. Raw generation and token IDs are retained. Leading LF characters are stripped, then only the first LF line is scored with standard normalized EM/F1 and predeclared dataset aliases. Controlled QA and TriviaQA are separate. Historical phrase matching is diagnostic only.',
        'The per-length descriptive gate remains ΔNLL ≤0.02 and EM retention ≥90% of raw-correct controlled questions at that same length. With zero raw-correct questions the result is unassessable. No thresholds or aliases were tuned on results.',
        'TTFT starts before socket setup and ends after the first generated token ID is available on CPU, including transfer, restore/tokenization, and remaining-input inference with CUDA synchronization. It excludes offline prefill/encoding, validation, and subsequent generation. Only the final required hidden state is projected to vocabulary for both text and cache methods. No network interface shaping is used.',
        'Offline prefill and representation encoding are measured separately. Their durations are once-per-context annotations repeated on rows for traceability, not costs incurred on each timed transfer. Adaptive calibration additionally records all-level encoding and preparation wall time, including raw QA/warmups/validation; it is not a minimal production cache-build estimate.',
        'Text wire format utf8_token_patch_v1 sends UTF-8 and explicit token-repair metadata for standalone subword boundaries or concatenated token segments. The receiver actually retokenizes and applies only the transmitted edits; all framing and token patches count in bytes. Patch counts are recorded. This is not an unmodified plain-text-only wire benchmark.',
        'The independent whole-text reference retains its original optimized single context+suffix forward. Both chunk controls instead execute the same dependent context prefill per chunk, then suffix-only inference. Pipeline benefit is primarily evaluated against chunk serial with identical messages, not attributed from a whole/chunk comparison.',
        f'Adaptive mode: {cfg["adaptive"]["mode"]}. Protocol v3 executes measured, serial paced socket transfers; it rejects trace_estimate for history feedback. SLO applies to TTFT; full answer latency is separate.',
        f'History throughput variant adaptive_history uses a fixed first-block estimate of {cfg["adaptive"]["initial_bandwidth_gbps"]} Gbps, then the previous completed full-message bytes * 8 / effective receive interval. The interval starts after receiver capacity/buffer readiness, includes pre-send pacing and socket overhead, and excludes queue-capacity and previous GPU waits. The history policy cannot access current or future trace values; those are executor inputs and posthoc diagnostics. Both adaptive variants share candidate costs, remaining budget, first-token reserve and fallback rules. This is a block-level trace simulation, not the complete paper scheduler.',
        'Each adaptive cached chunk comes from a full original-context prefill, so later-layer KV retains original-prefix dependence. Text fallback actually recomputes on the assembled prior KV with explicit positions. This is the repository heuristic, not the paper scheduler. Adaptive execution is serial. TTFT separately compares whole, chunk serial and bounded chunk pipeline; anchor/delta coding is absent.',
        'Byte ratios divide sums of actual serialized FP16 containers by sums of transmitted containers, including scales, CDFs, shapes and other metadata. Tensor-only ratios are separately named. Local socket target bandwidth is not measured throughput; see calibration.json and ttft.csv.',
        '## Paired quality by length', table(result['quality'], ['context_tokens', 'method', 'raw_serialized_over_method', 'paired_nll_delta', 'raw_em_correct', 'em_retention', 'acceptance']),
        '## Separate QA datasets', table(result['qa'], ['dataset', 'method', 'independent_questions_or_facts', 'conditions', 'mean_em', 'mean_f1']),
        '## TTFT', table(result['ttft'], ['context_tokens', 'target_bandwidth_gbps', 'method', 'mode', 'chunk_size', 'timing_observations', 'median_seconds', 'mean_actual_bandwidth_gbps', 'speedup_vs_text']),
        '## Paired pipeline comparison (same bytes)', table(result['pipeline_comparison'], ['context_tokens', 'target_bandwidth_gbps', 'method', 'chunk_size', 'paired_observations', 'median_paired_speedup', 'mean_pipeline_overlap_seconds']),
        'Pipeline speedup is serial/pipeline for each matched source, length, bandwidth, method, chunk size and repeat. Values below 1 mean slower pipeline. Chunk payload hashes and first tokens must match. Receiver capacity and send-permission waits, pacing, socket blocking, restoration, and overlap intervals are exported in ttft_timeline.csv / ttft_overlap.csv. All GPU restoration timestamps wait for CUDA completion.',
        '## Adaptive configurations', table(result['adaptive'], ['task', 'strategy', 'throughput_estimator', 'mode', 'fact_trace_conditions', 'timing_observations', 'median_ttft_seconds', 'slo_violation_rate', 'condition_mean_em', 'text_fallback_chunks']),
        '## Controlled mechanism ablations', table(result['ablation'], ['context_tokens', 'precision', 'entropy', 'mean_symbol_bits', 'mean_symbol_data_bytes', 'mean_complete_bytes', 'paired_nll_delta', 'mean_em', 'mean_encode_excluding_quantize_seconds', 'mean_decode_seconds']),
        'Each precision setting quantizes once; both entropy paths consume identical symbols/scales and share dequantization. Encoding timings exclude the separately recorded common quantization cost; complete sizes include CDF, stream lengths, scales, shapes and container overhead. uniform16 and uniform32 bracket layered2; differences in precision budget confound quality attribution. This small sweep shows size/quality tradeoffs, not isolated proof of layer allocation benefit.',
        '## Statistical interpretation', result['interval_note'],
        'Independent documents/questions/facts, derived length/position/trace conditions and timing repeats are separately counted. Cluster intervals keep all conditions and repeats of a source together. Small-sample P95 is descriptive. A completed matrix establishes execution coverage, not paper-scale reproduction or statistical equivalence.',
        '## Evidence',
        'config.json, selection.json, metadata.json, preflight.json, cache_checks.json, ttft_checks.json, calibration.json, records.jsonl, events.jsonl, records.sqlite3, manifest.json, failures.jsonl (if any), dependencies.txt, hardware.txt, source/, artifacts.json, and run.log preserve the protocol, data, code, environment, order, outputs and checkpoints.',
        f'Resume: `bash experiments/run.sh --resume {run}`. Integrity: `python3 -m experiments.suite --verify {run}`.',
        '## Figures', '![Size and NLL](figures/quality.png)', '![TTFT](figures/ttft.png)', '![Adaptive](figures/adaptive.png)',
        '![Paired pipeline speedup](figures/pipeline_speedup.png)', '![Receive and restore timeline](figures/receive_restore_timeline.png)',
        '![Mechanism size-quality sweep](figures/ablation.png)']
    (run / 'report.md').write_text('\n\n'.join(text) + '\n', encoding='utf-8')
    plots(run, result)
    extra_plots(run, result, journal.rows())
    return result


def plots(run, result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = run / 'figures'
    folder.mkdir(exist_ok=True)
    methods = list(dict.fromkeys(r['method'] for r in result['quality']))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for method in methods:
        rows = [r for r in result['quality'] if r['method'] == method]
        for axis, metric in zip(axes, ('raw_serialized_over_method', 'paired_nll_delta')):
            axis.plot([r['context_tokens'] for r in rows], [r[metric] for r in rows], marker='o', label=method)
            axis.set_xlabel('Context tokens')
    axes[0].set_ylabel('Serialized FP16 bytes / transmitted bytes')
    axes[1].set_ylabel('Paired complete-continuation NLL delta')
    axes[1].axhline(.02, color='k', linestyle='--', linewidth=.8)
    axes[0].legend(fontsize=8)
    _savefig(fig, folder, 'quality', plt)
    lengths = sorted(set(r['context_tokens'] for r in result['ttft']))
    fig, axes = plt.subplots(1, len(lengths), figsize=(5 * len(lengths), 4), squeeze=False, constrained_layout=True)
    for axis, length in zip(axes[0], lengths):
        variants = dict.fromkeys((r['method'], r['mode'], r['chunk_size']) for r in result['ttft'] if r['context_tokens'] == length)
        for method, mode, size in variants:
            rows = [r for r in result['ttft'] if (r['method'], r['mode'], r['chunk_size'], r['context_tokens']) == (method, mode, size, length)]
            axis.plot([r['target_bandwidth_gbps'] for r in rows], [r['median_seconds'] for r in rows], marker='o', label=f'{method}/{mode}/c{size}')
        axis.set(xscale='log', yscale='log', xlabel='Target Gbps (actual in CSV)', ylabel='Measured TTFT seconds', title=f'{length} tokens')
        axis.legend(fontsize=5, ncol=2)
    _savefig(fig, folder, 'ttft', plt)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    names = list(dict.fromkeys(r['task'] for r in result['adaptive']))
    for strategy in dict.fromkeys(r['strategy'] for r in result['adaptive']):
        rows = [r for r in result['adaptive'] if r['strategy'] == strategy]
        for axis, metric in zip(axes, ('median_ttft_seconds', 'slo_violation_rate')):
            axis.plot(range(len(names)), [r[metric] for r in rows], marker='o', label=strategy)
            axis.set_xticks(range(len(names)), [n.replace('adaptive-', '') for n in names], rotation=15, fontsize=8)
    axes[0].set_ylabel('TTFT seconds (mode in CSV)')
    axes[1].set_ylabel('SLO violation fraction')
    axes[0].legend(fontsize=8)
    _savefig(fig, folder, 'adaptive', plt)


def _savefig(fig, folder, name, plt):
    for extension in ('png', 'svg', 'pdf'):
        fig.savefig(folder / f'{name}.{extension}', dpi=150)
    plt.close(fig)
