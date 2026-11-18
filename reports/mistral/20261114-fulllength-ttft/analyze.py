"""Stdlib-only TTFT model from measured timings. Usage: python3 analyze.py

serial  = bytes*8/bandwidth + decode + first token on the restored KV (nothing overlaps);
overlap = max(bytes*8/bandwidth, decode) + first token: ideal chunk pipelining, a lower bound.
text    = text bytes*8/bandwidth + prefill of conversation + question.
Quality columns come from the earlier full-length runs on the same IDs (question KV computed fresh).
"""
import json, statistics as st
from pathlib import Path
here = Path(__file__).parent
load = lambda p: [json.loads(l) for l in p.open()]
rows = load(here / 'records.jsonl')
BANDWIDTHS = (0.5, 1, 3, 10)  # Gbps
METHODS = ('text', 'raw', 'uniform8', 'uniform4', 'cachegen3', 'cachegen2', 'cachegen1', 'ac_b12')
by = {m: {r['source_id']: r for r in rows if r['method'] == m} for m in METHODS}
ids = sorted(by['text'])

quality = {}
for path in ('20261102-fulllength-qa', '20261102-fulllength-ac'):
    for r in load(here.parent / path / 'records.jsonl'):
        if r['arm'] == 'clean_question' and r['source_id'] in ids:
            quality.setdefault(r['method'], {})[r['source_id']] = r

def ttft(r, bw, mode):
    transfer = r['bytes'] * 8 / (bw * 1e9)
    if r['method'] == 'text':
        return transfer + r['first_token_seconds']
    if mode == 'serial':
        return transfer + r['decode_seconds'] + r['first_token_seconds']
    return max(transfer, r['decode_seconds']) + r['first_token_seconds']

out = {'ids': ids, 'methods': {}, 'speedup': {}}
for m in METHODS:
    v = by[m]
    q = quality.get(m, {})
    out['methods'][m] = dict(
        n=len(v), mean_mb=st.mean(r['bytes'] for r in v.values()) / 1e6,
        decode_median_s=st.median(r['decode_seconds'] for r in v.values()),
        first_token_median_s=st.median(r['first_token_seconds'] for r in v.values()),
        first_token_matches=sum(r['matches_direct'] for r in v.values()),
        includes=sum(q[i]['includes_answer'] for i in ids) if len(q) == len(ids) else None,
        identical=sum(q[i]['identical_to_raw'] for i in ids) if len(q) == len(ids) else None,
        ttft_median_s={f'{bw}': {mode: st.median(ttft(r, bw, mode) for r in v.values())
                                 for mode in ('serial', 'overlap')} for bw in BANDWIDTHS})
for a in ('cachegen1', 'ac_b12', 'cachegen2'):
    for b in ('text', 'uniform8', 'uniform4', 'raw'):
        for mode in ('serial', 'overlap'):
            out['speedup'][f'{a} vs {b} ({mode})'] = {
                f'{bw}': round(st.median(ttft(by[b][i], bw, mode) / ttft(by[a][i], bw, mode) for i in ids), 3)
                for bw in BANDWIDTHS}
spread = [max(r['decode_runs']) / min(r['decode_runs']) for m in METHODS[1:] for r in by[m].values()]
out['timing_spread_max_over_min'] = dict(median=st.median(spread), max=max(spread))
out['peak_gpu_gib'] = max(r.get('peak_gpu_bytes', 0) for r in rows) / 2**30
(here / 'summary.json').write_text(json.dumps(out, indent=1))
for m, r in out['methods'].items():
    t = ' '.join(f"{bw}G:{r['ttft_median_s'][f'{bw}']['serial']:.2f}/{r['ttft_median_s'][f'{bw}']['overlap']:.2f}" for bw in BANDWIDTHS)
    print(f"{m:10s} {r['mean_mb']:8.1f}MB dec={r['decode_median_s']:.3f} first={r['first_token_median_s']:.3f} "
          f"match={r['first_token_matches']} acc={r['includes']} same={r['identical']} | {t}")
for k, v in out['speedup'].items():
    print(k, v)
print(out['timing_spread_max_over_min'], round(out['peak_gpu_gib'], 1))
