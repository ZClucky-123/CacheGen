#!/usr/bin/env python3
"""复算论文差异记录中的全部数字。只读 stage1/stage2 下载结果，不加载模型，只依赖标准库。"""
import collections
import json
import random
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
S1 = HERE.parent / '20261029-paper-core-stage1'
S2 = HERE.parent / '20261029-paper-core-stage2'
CORE, NODELTA, NOLAYER = 'core-r8-original', 'no-delta-r10', 'no-layer-r8-original'
LENGTHS = (1024, 2048, 4096)


def jsonl(path):
    return [json.loads(line) for line in Path(path).open()]


def ci(values, rounds=5000, seed=0):
    rng = random.Random(seed)
    means = sorted(st.mean(rng.choices(values, k=len(values))) for _ in range(rounds))
    return [means[int(.025 * rounds)], means[int(.975 * rounds)]]


def compression(out):
    rows = jsonl(S1 / 'test/records.jsonl')
    idx = {(r['source_id'], r['context_tokens'], r['method']): r for r in rows}
    methods = ('uniform8', 'uniform4', 'cachegen1', 'cachegen2', 'cachegen3', CORE, NODELTA, NOLAYER)
    table = {}
    for n in LENGTHS:
        table[n] = {}
        for m in methods:
            group = [idx[i, n, m] for i in range(20)]
            nll = [r['delta_nll'] for r in group]
            real = [r['actual_bytes_relative_uniform8'] for r in group if r.get('actual_bytes_relative_uniform8')]
            table[n][m] = {'size_vs_uniform8': st.mean(r['bytes_relative_uniform8'] for r in group),
                           'size_kind': group[0].get('size_kind'),
                           'real_coded_size_vs_uniform8': st.mean(real) if real else None,
                           'real_coded_docs': len(real), 'delta_nll': st.mean(nll), 'delta_nll_ci': ci(nll)}
    out['test_by_length'] = table
    pairs = {}
    for a, b in ((CORE, NODELTA), (CORE, NOLAYER), (CORE, 'cachegen2'), (CORE, 'cachegen3'), (CORE, 'uniform4'),
                 ('cachegen2', 'uniform4'), ('cachegen3', 'uniform4')):
        for n in LENGTHS:
            diff = [idx[i, n, a]['delta_nll'] - idx[i, n, b]['delta_nll'] for i in range(20)]
            pairs[f'{a} vs {b} @{n}'] = {
                'size_ratio': table[n][a]['size_vs_uniform8'] / table[n][b]['size_vs_uniform8'],
                'paired_delta_nll': st.mean(diff), 'ci': ci(diff)}
    out['paired'] = pairs
    selected = json.load((S1 / 'test/frozen_selection.json').open())
    out['selected'] = selected['selected']
    by = {r['method']: r for r in selected['candidates']}
    ranges = ('6', '8', '10', '12', '15', '20', '25', '31', '50', '127')
    out['same_range_size_ratio'] = {
        v: {'core/no-delta': by[f'core-r{v}-original']['estimated_bytes'] / by[f'no-delta-r{v}']['estimated_bytes'],
            'core/no-layer': by[f'core-r{v}-original']['estimated_bytes'] / by[f'no-layer-r{v}-original']['estimated_bytes']}
        for v in ranges}
    real = [r for r in rows if r['method'] == CORE and r['context_tokens'] == 4096 and r.get('actual_bytes')]
    out['core_4k_bytes_mb'] = {k: st.mean(r[k] for r in real) / 1e6 for k in (
        'actual_bytes', 'actual_stream_bytes', 'cross_entropy_bytes', 'segment_lengths_bytes',
        'scale_bytes', 'estimated_bytes')}
    out['core_estimate_relative_error'] = st.mean(r['estimate_relative_error'] for r in real)
    qa = json.load((S1 / 'test/qa_selection.json').open())
    answers = {int(k): v['answers'][0].lower() for k, v in qa.items()}
    out['qa_4k'] = {}
    for m in ('raw',) + methods:
        group = [idx[i, 4096, m] for i in range(20)]
        out['qa_4k'][m] = {
            'contains_answer': sum(answers[r['source_id']] in r['qa']['prediction'].lower() for r in group),
            'identical_to_raw': sum(r['qa']['prediction'] == idx[r['source_id'], 4096, 'raw']['qa']['prediction'] for r in group),
            'em': st.mean(r['qa']['em'] for r in group), 'f1': st.mean(r['qa']['f1'] for r in group)}
    out['qa_example'] = {'prediction': idx[0, 4096, 'raw']['qa']['prediction'], 'answer': qa['0']['answers'][0],
                         'em': idx[0, 4096, 'raw']['qa']['em'], 'f1': idx[0, 4096, 'raw']['qa']['f1']}


def insights(out):
    rows = jsonl(S1 / 'insights/locality.jsonl')
    raw = [r for r in rows if r['representation'] == 'raw']
    locality = {}
    for name, group in (('anchor_delta', [r for r in rows if r['representation'] == 'anchor_delta']),
                        ('lag1', [r for r in rows if r.get('lag') == 1]), ('lag9', [r for r in rows if r.get('lag') == 9])):
        ratios = lambda s: st.mean(st.mean(b / a for a, b in zip(x['variance_by_kv_layer'][s], y['variance_by_kv_layer'][s]))
                                   for x, y in zip(raw, group))
        locality[name] = {'K_variance_ratio': ratios(slice(0, 32)), 'V_variance_ratio': ratios(slice(32, 64))}
    out['locality'] = locality
    by = collections.defaultdict(dict)
    for r in jsonl(S1 / 'insights/sensitivity.jsonl'):
        by[r['source_id']][r['layer_group']] = r['delta_nll']
    sens = {f'group{g}': {'delta_nll': st.mean(v[g] for v in by.values()), 'ci': ci([v[g] for v in by.values()])} for g in range(3)}
    for a, b in ((0, 1), (0, 2), (1, 2)):
        diff = [v[a] - v[b] for v in by.values()]
        sens[f'group{a}-group{b}'] = {'mean': st.mean(diff), 'ci': ci(diff), f'group{a}_lower_docs': sum(d < 0 for d in diff), 'docs': len(diff)}
    out['sensitivity'] = sens
    grouping = json.load((S1 / 'insights/grouping.json').open())
    out['grouping_bits_per_symbol'] = grouping['bits_per_symbol']


def network(out):
    rows = jsonl(S2 / 'ttft/records.jsonl')
    cells = collections.defaultdict(list)
    for r in rows:
        cells[r['method'], r['mode'], r['target_bandwidth_gbps']].append(r)
    ttft = {}
    for b in (.5, 1, 3, 10):
        ttft[str(b)] = {}
        for m, mode in ((CORE, 'chunk_pipeline'), (CORE, 'chunk_serial'), (CORE, 'whole'), ('uniform4', 'chunk_pipeline'),
                        ('uniform8', 'whole'), ('uniform8', 'chunk_pipeline'), ('text', 'chunk_pipeline'), ('text', 'whole')):
            group = cells[m, mode, b]
            ttft[str(b)][f'{m}/{mode}'] = {'median_ttft': st.median(r['ttft_wall_seconds'] for r in group),
                                           'median_actual_gbps': st.median(r['actual_bandwidth_gbps'] for r in group)}
    out['ttft_4k'] = ttft
    rows = jsonl(S2 / 'adaptive/records.jsonl')
    violations = collections.Counter()
    runs = collections.Counter()
    bytes_used = collections.defaultdict(list)
    choices = collections.Counter()
    for r in rows:
        key = f"SLO {r['slo_seconds']} / {r['strategy']}"
        runs[key] += 1
        violations[key] += r['slo_violation']
        bytes_used[key].append(r['total_bytes_including_probe'])
        if r['strategy'] == 'history':
            for d in r['decisions']:
                choices[f"SLO {r['slo_seconds']} trace {r['trace_id']} {'text' if d['method'] == 'text' else 'KV'}"] += 1
    out['adaptive'] = {k: {'violations': violations[k], 'runs': runs[k], 'mean_mb': st.mean(bytes_used[k]) / 1e6} for k in sorted(runs)}
    out['history_choices'] = dict(sorted(choices.items()))


def main():
    out = {}
    compression(out)
    insights(out)
    network(out)
    (HERE / 'recomputed.json').write_text(json.dumps(out, indent=1, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
