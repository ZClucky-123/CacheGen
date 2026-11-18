"""Stdlib-only summary of records.jsonl. Usage: python3 analyze.py"""
import json, math, collections, statistics as st
from pathlib import Path
here = Path(__file__).parent
rows = [json.loads(l) for l in (here / 'records.jsonl').open()]
METHODS = ('raw', 'uniform8', 'uniform4', 'uniform3', 'cachegen1', 'cachegen2', 'cachegen3')

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z*z/n; c = (p + z*z/(2*n)) / d
    h = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return max(0, c-h), min(1, c+h)

def exact_mcnemar(b, c):
    n = b + c
    if not n: return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c)+1)) / 2**n)

out = {}
for arm in ('author', 'clean_question'):
    for name, ids in (('all', range(50)), ('id0-19', range(20)), ('id20-49', range(20, 50))):
        sel = {m: {r['source_id']: r for r in rows if r['arm'] == arm and r['method'] == m and r['source_id'] in ids} for m in METHODS}
        base = sel['uniform8']; t = {}
        for m in METHODS:
            v = sel[m]; n = len(v); k = sum(x['includes_answer'] for x in v.values())
            k1 = sum(x['includes_answer_first_line'] for x in v.values())
            t[m] = dict(n=n, includes=k, rate=k/n, wilson95=wilson(k, n), first_line=k1,
                        identical=sum(x['identical_to_raw'] for x in v.values()),
                        other_topic=sum(x['first_line_names_other_topic'] for x in v.values()),
                        mean_mb=st.mean(x['bytes'] for x in v.values())/1e6,
                        vs_8bit=st.mean(v[i]['bytes']/base[i]['bytes'] for i in v),
                        mse=st.mean(x['mse'] for x in v.values()),
                        tokens=st.median(x['kv_tokens'] for x in v.values()))
        out[f'{arm}/{name}'] = t
pairs = {}
for arm in ('author', 'clean_question'):
    get = lambda m: {r['source_id']: r['includes_answer'] for r in rows if r['arm'] == arm and r['method'] == m}
    for a, b in (('cachegen1','uniform3'),('cachegen2','uniform3'),('cachegen1','uniform4'),('cachegen2','uniform4'),('uniform4','uniform3')):
        A, B = get(a), get(b)
        x = sum(A[i] and not B[i] for i in A); y = sum(B[i] and not A[i] for i in A)
        pairs[f'{arm}: {a} vs {b}'] = dict(a_only=x, b_only=y, exact_p=exact_mcnemar(x, y))
gap = {}
for m in METHODS:
    A = {r['source_id']: r['includes_answer'] for r in rows if r['arm']=='author' and r['method']==m}
    C = {r['source_id']: r['includes_answer'] for r in rows if r['arm']=='clean_question' and r['method']==m}
    x = sum(A[i] and not C[i] for i in A); y = sum(C[i] and not A[i] for i in A)
    gap[m] = dict(author_only=x, clean_only=y, exact_p=exact_mcnemar(x, y))
tok = sorted({(r['source_id'], r['kv_tokens']) for r in rows if r['arm']=='author'})
summary = dict(by_group=out, paired=pairs, author_vs_clean=gap, records=len(rows),
               token_min=min(t for _, t in tok), token_max=max(t for _, t in tok),
               peak_gpu_gib=max(r['peak_gpu_bytes'] for r in rows)/2**30)
(here / 'summary.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False))
for g in ('author/all', 'clean_question/all', 'author/id0-19', 'clean_question/id0-19'):
    print(g)
    for m, r in out[g].items():
        print(f"  {m:10s} {r['includes']}/{r['n']} [{r['wilson95'][0]:.3f},{r['wilson95'][1]:.3f}] first={r['first_line']} same={r['identical']} other={r['other_topic']} MB={r['mean_mb']:.1f} vs8={r['vs_8bit']:.3f} mse={r['mse']:.3g}")
for k, v in pairs.items(): print(k, v)
print(gap); print(summary['records'], summary['token_min'], summary['token_max'], round(summary['peak_gpu_gib'],1))
