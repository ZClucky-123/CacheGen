"""Protocol-v3 dimensions, paired comparisons and exported receive timelines."""
from collections import defaultdict
import statistics as stats


def pipeline_pairs(rows):
    pairs = defaultdict(dict)
    for r in rows:
        if r['kind'] == 'ttft' and r['mode'] in ('chunk_serial', 'chunk_pipeline'):
            key = (r['case_id'], r['context_tokens'], r['bandwidth_gbps'], r['method'],
                   r['chunk_size'], r['queue_capacity'], r['repetition'])
            if r['mode'] in pairs[key]:
                raise ValueError('Duplicate TTFT pair member')
            pairs[key][r['mode']] = r
    result = []
    for key, modes in pairs.items():
        if set(modes) != {'chunk_serial', 'chunk_pipeline'}:
            raise ValueError('Incomplete serial/pipeline pair')
        serial, pipeline = modes['chunk_serial'], modes['chunk_pipeline']
        if serial['prepared_payload_sha256'] != pipeline['prepared_payload_sha256'] or serial['bytes'] != pipeline['bytes']:
            raise ValueError('Paired modes did not reuse identical messages')
        if serial['first_token_id'] != pipeline['first_token_id']:
            raise ValueError('Paired modes generated different first tokens')
        result.append({'case_id': key[0], 'context_tokens': key[1], 'target_bandwidth_gbps': key[2],
            'method': key[3], 'chunk_size': key[4], 'queue_capacity': key[5], 'repetition': key[6],
            'source_id': serial['source_id'], 'bytes': serial['bytes'],
            'serial_seconds': serial['ttft_wall_seconds'], 'pipeline_seconds': pipeline['ttft_wall_seconds'],
            'serial_over_pipeline': serial['ttft_wall_seconds'] / pipeline['ttft_wall_seconds'],
            'pipeline_overlap_seconds': pipeline['overlap_seconds']})
    return result


def pipeline_summary(pairs, seed):
    from experiments.analysis import cluster_interval
    groups = defaultdict(list)
    for r in pairs:
        groups[(r['context_tokens'], r['target_bandwidth_gbps'], r['method'], r['chunk_size'], r['queue_capacity'])].append(r)
    return [{'context_tokens': key[0], 'target_bandwidth_gbps': key[1], 'method': key[2], 'chunk_size': key[3],
        'queue_capacity': key[4], 'paired_observations': len(group),
        'independent_source_documents': len({r['source_id'] for r in group}),
        'median_paired_speedup': stats.median(r['serial_over_pipeline'] for r in group),
        'mean_paired_speedup': stats.mean(r['serial_over_pipeline'] for r in group),
        'mean_speedup_source_cluster_ci95': cluster_interval([(r['source_id'], r['serial_over_pipeline']) for r in group], seed),
        'mean_pipeline_overlap_seconds': stats.mean(r['pipeline_overlap_seconds'] for r in group)}
        for key, group in sorted(groups.items())]


def ablation_summary(rows):
    natural_raw = {r['case_id']: r for r in rows if r['kind'] == 'quality' and r['method'] == 'raw'}
    qa_raw = {r['case_id']: r for r in rows if r['kind'] == 'controlled' and r['method'] == 'raw'}
    groups, pairs = defaultdict(list), defaultdict(dict)
    for r in rows:
        if r['kind'] == 'ablation':
            groups[(r['context_tokens'], r['precision'], r['entropy'])].append(r)
            pairs[(r['evaluation_kind'], r['case_id'], r['precision'])][r['entropy']] = r
    pair_rows = []
    for key, modes in pairs.items():
        if set(modes) != {'packed', 'arithmetic'}:
            raise ValueError('Incomplete entropy pair')
        packed, arithmetic = modes['packed'], modes['arithmetic']
        if any(packed[f] != arithmetic[f] for f in ('symbols_sha256', 'scales_sha256', 'bins')):
            raise ValueError('Entropy pair changed quantization')
        if packed.get('token_nll') != arithmetic.get('token_nll') or packed.get('generated_ids') != arithmetic.get('generated_ids'):
            raise ValueError('Identical symbols produced different NLL/QA')
        pair_rows.append({'evaluation_kind': key[0], 'case_id': key[1], 'precision': key[2],
            'packed_symbol_bytes': packed['symbol_data_bytes'], 'arithmetic_symbol_bytes': arithmetic['symbol_data_bytes'],
            'packed_complete_bytes': packed['bytes'], 'arithmetic_complete_bytes': arithmetic['bytes'],
            'complete_packed_over_arithmetic': packed['bytes'] / arithmetic['bytes'],
            'symbols_and_scales_identical': True, 'quality_identical': True})
    summary = []
    for (length, precision, entropy), group in sorted(groups.items()):
        natural = [r for r in group if r['evaluation_kind'] == 'quality']
        qa = [r for r in group if r['evaluation_kind'] == 'controlled']
        raw_correct = [r for r in qa if qa_raw[r['case_id']]['em'] == 1]
        summary.append({'context_tokens': length, 'precision': precision, 'entropy': entropy,
            'natural_conditions': len(natural), 'controlled_conditions': len(qa),
            'mean_symbol_bits': group[0]['mean_symbol_bits'],
            'mean_symbol_data_bytes': stats.mean(r['symbol_data_bytes'] for r in natural),
            'mean_complete_bytes': stats.mean(r['bytes'] for r in natural),
            'mean_cdf_tensor_bytes': stats.mean(r['cdf_tensor_bytes'] for r in natural),
            'mean_nll': stats.mean(r['nll'] for r in natural),
            'paired_nll_delta': stats.mean(r['nll'] - natural_raw[r['case_id']]['nll'] for r in natural),
            'mean_em': stats.mean(r['em'] for r in qa), 'mean_f1': stats.mean(r['f1'] for r in qa),
            'raw_em_correct': len(raw_correct), 'em_retention': stats.mean(r['em'] for r in raw_correct) if raw_correct else None,
            'mean_shared_quantize_seconds': stats.mean(r['quantize_seconds'] for r in natural),
            'mean_encode_excluding_quantize_seconds': stats.mean(r['offline_encode_seconds'] for r in natural),
            'mean_decode_seconds': stats.mean(r['decode_seconds'] for r in natural)})
    return summary, pair_rows


def timeline_rows(rows):
    result, overlaps = [], []
    for r in rows:
        if r['kind'] != 'ttft':
            continue
        common = {k: r[k] for k in ('record_id', 'case_id', 'context_tokens', 'method', 'mode', 'chunk_size',
                                     'queue_capacity', 'bandwidth_gbps', 'repetition')}
        for c in r['chunks']:
            result.append({**common, **{k: v for k, v in c.items() if k not in ('shape', 'internal_segment_tokens')}})
        overlaps.extend({**common, **interval} for interval in r['overlap_intervals'])
    return result, overlaps


def extra_plots(run, result, rows):
    import matplotlib.pyplot as plt
    from experiments.analysis import _savefig
    folder = run / 'figures'
    pipeline = result['pipeline_comparison']
    if pipeline:
        fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
        labels = [f'{r["method"]}\nL{r["context_tokens"]} {r["target_bandwidth_gbps"]}G c{r["chunk_size"]}' for r in pipeline]
        ax.bar(range(len(pipeline)), [r['median_paired_speedup'] for r in pipeline])
        ax.set_xticks(range(len(pipeline)), labels, rotation=70, fontsize=6)
        ax.set_ylabel('Paired serial TTFT / pipeline TTFT')
        ax.axhline(1., color='k', linestyle='--')
        _savefig(fig, folder, 'pipeline_speedup', plt)
        # One actual matched condition, two modes, same encoded bytes. Every
        # observation's full timeline is exported in ttft_timeline.csv.
        examples = [r for r in rows if r['kind'] == 'ttft' and r['mode'] == 'chunk_serial']
        example = max(examples, key=lambda r: (len(r['chunks']), r['method'] == 'cachegen2'))
        match = next(r for r in rows if r['kind'] == 'ttft' and r['mode'] == 'chunk_pipeline' and all(
            r[k] == example[k] for k in ('case_id', 'method', 'chunk_size', 'bandwidth_gbps', 'repetition')))
        fig, axes = plt.subplots(2, 1, figsize=(10, 5), constrained_layout=True)
        from matplotlib.patches import Patch
        for ax, row in zip(axes, (example, match)):
            for c in row['chunks']:
                ax.broken_barh([(c['receive_start'], c['data_available'] - c['receive_start'])], (c['chunk_index'] * 2, .8), facecolors='tab:blue')
                ax.broken_barh([(c['restore_start'], c['restore_end'] - c['restore_start'])], (c['chunk_index'] * 2 + .9, .8), facecolors='tab:orange')
            ax.set(title=f'{row["mode"]}: {row["method"]}, {row["context_tokens"]} tokens, {row["bandwidth_gbps"]} Gbps',
                   xlabel='Seconds from before socket creation', ylabel='Chunk (receive / restore)')
            ax.set_yticks([c['chunk_index'] * 2 + .8 for c in row['chunks']],
                          [str(c['chunk_index']) for c in row['chunks']])
            ax.set_xlim(0, max(example['ttft_wall_seconds'], match['ttft_wall_seconds']) * 1.02)
            first_line = ax.axvline(row['ttft_wall_seconds'], color='k', linestyle=':', label='First CPU token ID')
            ax.legend(handles=[Patch(color='tab:blue', label='Receive (incl. pacing)'),
                               Patch(color='tab:orange', label='Restore (CUDA complete)'), first_line], fontsize=7)
        _savefig(fig, folder, 'receive_restore_timeline', plt)
    if result['ablation']:
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        for r in result['ablation']:
            ax.scatter(r['mean_complete_bytes'] / 2**20, r['paired_nll_delta'],
                       marker='o' if r['entropy'] == 'packed' else '^',
                       label=f'{r["precision"]}/{r["entropy"]}/L{r["context_tokens"]}')
        ax.set(xlabel='Complete serialized MiB (including CDF/metadata)', ylabel='Paired NLL delta vs raw')
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
        _savefig(fig, folder, 'ablation', plt)
