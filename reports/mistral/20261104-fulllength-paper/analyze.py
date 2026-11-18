"""Stdlib-only summary of records.jsonl against the earlier runs on the same IDs. Usage: python3 analyze.py"""
import json, math, statistics as st
from pathlib import Path
here = Path(__file__).parent
load = lambda p: [json.loads(l) for l in p.open()]
rows = load(here / 'records.jsonl')
arm = 'clean_question'
ids = range(20)
qa = load(here.parent / '20261102-fulllength-qa/records.jsonl')
ac = load(here.parent / '20261102-fulllength-ac/records.jsonl')

def table(data, m):
    return {r['source_id']: r for r in data if r['arm'] == arm and r['method'] == m and r['source_id'] in ids}
def wilson(k, n, z=1.96):
    p = k/n; d = 1+z*z/n; c = (p+z*z/(2*n))/d
    h = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return round(max(0, c-h), 3), round(min(1, c+h), 3)
def mcnemar(b, c):
    n = b+c
    return 1.0 if not n else min(1.0, 2*sum(math.comb(n, i) for i in range(min(b, c)+1))/2**n)

METHODS = {}
for m in ('raw', 'paper-r2', 'paper-r3', 'paper-r5', 'paper-r7.5', 'nolayer-r3', 'nolayer-r5', 'nodelta-r3', 'nodelta-r5', 'nodelta-r7.5'):
    METHODS[m] = table(rows, m)
for m in ('uniform8', 'uniform4', 'uniform3', 'cachegen1', 'cachegen2', 'cachegen3'):
    METHODS['earlier:' + m] = table(qa, m)
for m in ('uniform8_ac', 'ac_b16', 'ac_b12', 'ac_b10', 'ac_b8', 'layer_a', 'layer_b'):
    METHODS['earlier:' + m] = table(ac, m)
u8 = METHODS['earlier:uniform8']
out = {}
for m, v in METHODS.items():
    k = sum(x['includes_answer'] for x in v.values())
    out[m] = dict(n=len(v), includes=k, wilson95=wilson(k, len(v)),
                  identical=sum(x['identical_to_raw'] for x in v.values()),
                  eos=sum(x['stop_reason'] == 'eos' for x in v.values()),
                  other_topic=sum(x.get('first_line_names_other_topic', 0) for x in v.values()),
                  mean_mb=st.mean(x['bytes'] for x in v.values())/1e6,
                  vs_8bit=st.mean(v[i]['bytes']/u8[i]['bytes'] for i in v),
                  mse=st.mean(x['mse'] for x in v.values()),
                  stream_fraction=(st.mean(x['stream_bytes']/x['bytes'] for x in v.values()) if 'stream_bytes' in next(iter(v.values())) else None),
                  escapes=sum(x.get('escape_count', 0) for x in v.values()))
PAIRS = [('paper-r3', 'nolayer-r3'), ('paper-r5', 'nolayer-r5'), ('paper-r3', 'nodelta-r3'), ('paper-r5', 'nodelta-r5'),
         ('paper-r7.5', 'nodelta-r7.5'), ('paper-r3', 'nodelta-r5'), ('paper-r5', 'nodelta-r7.5'), ('paper-r2', 'nodelta-r3'),
         ('paper-r5', 'earlier:ac_b12'), ('paper-r3', 'earlier:ac_b12'), ('paper-r2', 'earlier:ac_b12'), ('nodelta-r5', 'earlier:ac_b12'),
         ('paper-r5', 'earlier:cachegen2'), ('paper-r7.5', 'earlier:cachegen3'), ('nolayer-r3', 'earlier:cachegen1'), ('paper-r3', 'earlier:cachegen1')]
pairs = {}
for a, b in PAIRS:
    A, B = METHODS[a], METHODS[b]
    d = {}
    for key, f in (('includes', lambda r: r['includes_answer']), ('identical', lambda r: r['identical_to_raw']), ('eos', lambda r: r['stop_reason'] == 'eos')):
        x = sum(bool(f(A[i])) and not f(B[i]) for i in A); y = sum(bool(f(B[i])) and not f(A[i]) for i in A)
        d[key] = dict(a_only=x, b_only=y, exact_p=round(mcnemar(x, y), 4))
    d['size_ratio'] = round(st.mean(A[i]['bytes']/B[i]['bytes'] for i in A), 3)
    pairs[f'{a} vs {b}'] = d
raw_old = METHODS['earlier:uniform8']
checks = dict(
    raw_same_as_earlier_raw=sum(METHODS['raw'][i]['prediction'] == r['prediction'] for i, r in table(qa, 'raw').items()),
    shared_table_mb=rows[1]['shared_table_bytes']/1e6)
summary = dict(by_method=out, paired=pairs, checks=checks, records=len(rows),
               peak_gpu_gib=max(r['peak_gpu_bytes'] for r in rows)/2**30)
(here / 'summary.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False))
for m, r in sorted(out.items(), key=lambda kv: kv[1]['mean_mb']):
    print(f"{m:22s} {r['mean_mb']:7.1f}MB {r['vs_8bit']:.3f} acc={r['includes']:2}/{r['n']} same={r['identical']:2} eos={r['eos']:2} other={r['other_topic']:2} mse={r['mse']:.3f} stream={r['stream_fraction']} esc={r['escapes']}")
for k, v in pairs.items(): print(k, v)
print(checks, summary['peak_gpu_gib'])
