"""Stdlib-only summary of records.jsonl plus the earlier uncoded baselines. Usage: python3 analyze.py"""
import json, math, statistics as st
from pathlib import Path
here = Path(__file__).parent
rows = [json.loads(l) for l in (here / 'records.jsonl').open()]
old = [json.loads(l) for l in (here.parent / '20261102-fulllength-qa/records.jsonl').open()]
ORDER = ('raw', 'uniform8_ac', 'ac_b16', 'cachegen1', 'ac_b12', 'layer_a', 'ac_b10', 'layer_b', 'layer_c', 'ac_b8', 'layer_d', 'ac_b6')

def wilson(k, n, z=1.96):
    p = k/n; d = 1+z*z/n; c = (p+z*z/(2*n))/d
    h = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return round(max(0, c-h), 3), round(min(1, c+h), 3)
def mcnemar(b, c):
    n = b+c
    return 1.0 if not n else min(1.0, 2*sum(math.comb(n, i) for i in range(min(b, c)+1))/2**n)

def table(data, arm, m, ids=range(50)):
    return {r['source_id']: r for r in data if r['arm'] == arm and r['method'] == m and r['source_id'] in ids}
out = {}
for arm in ('author', 'clean_question'):
    u8 = table(old, arm, 'uniform8')
    for name, ids in (('all', range(50)), ('id0-19', range(20)), ('id20-49', range(20, 50))):
        t = {}
        for m in ORDER:
            v = table(rows, arm, m, ids); k = sum(x['includes_answer'] for x in v.values())
            t[m] = dict(n=len(v), includes=k, wilson95=wilson(k, len(v)),
                        first_line=sum(x['includes_answer_first_line'] for x in v.values()),
                        other_topic=sum(x['first_line_names_other_topic'] for x in v.values()),
                        identical=sum(x['identical_to_raw'] for x in v.values()),
                        mean_mb=st.mean(x['bytes'] for x in v.values())/1e6,
                        vs_8bit=st.mean(v[i]['bytes']/u8[i]['bytes'] for i in v),
                        mse=st.mean(x['mse'] for x in v.values()))
        out[f'{arm}/{name}'] = t
pairs = {}
for arm in ('author', 'clean_question'):
    for a, b in (('cachegen1','ac_b16'),('cachegen1','ac_b12'),('layer_a','ac_b12'),('layer_a','ac_b10'),('layer_b','ac_b10'),('layer_c','ac_b10'),('layer_c','ac_b8'),('layer_d','ac_b8'),('layer_d','ac_b6')):
        A, B = table(rows, arm, a), table(rows, arm, b)
        x = sum(A[i]['includes_answer'] and not B[i]['includes_answer'] for i in A)
        y = sum(B[i]['includes_answer'] and not A[i]['includes_answer'] for i in A)
        pairs[f'{arm}: {a} vs {b}'] = dict(a_only=x, b_only=y, exact_p=mcnemar(x, y),
            size_ratio=st.mean(A[i]['bytes']/B[i]['bytes'] for i in A))
checks = {}
for arm in ('author', 'clean_question'):
    for new, base in (('ac_b16','uniform4'),('ac_b8','uniform3'),('uniform8_ac','uniform8'),('cachegen1','cachegen1')):
        A, B = table(rows, arm, new), table(old, arm, base)
        checks[f'{arm}: {new} vs {base}'] = dict(
            same_prediction=sum(A[i]['prediction'] == B[i]['prediction'] for i in A),
            max_rel_mse_diff=max(abs(A[i]['mse']-B[i]['mse'])/max(B[i]['mse'], 1e-12) for i in A),
            size_ratio=st.mean(A[i]['bytes']/B[i]['bytes'] for i in A),
            acc=(sum(A[i]['includes_answer'] for i in A), sum(B[i]['includes_answer'] for i in B)))
summary = dict(by_group=out, paired=pairs, against_earlier_run=checks, records=len(rows),
               peak_gpu_gib=max(r['peak_gpu_bytes'] for r in rows)/2**30)
(here / 'summary.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False))
for g in ('author/all', 'clean_question/all', 'author/id0-19', 'clean_question/id0-19'):
    print(g)
    for m, r in out[g].items():
        print(f"  {m:12s} {r['includes']}/{r['n']} {r['wilson95']} first={r['first_line']} other={r['other_topic']} same={r['identical']} MB={r['mean_mb']:.1f} vs8={r['vs_8bit']:.3f} mse={r['mse']:.3g}")
for k, v in pairs.items(): print(k, v)
for k, v in checks.items(): print(k, v)
print(summary['records'], round(summary['peak_gpu_gib'], 1))
