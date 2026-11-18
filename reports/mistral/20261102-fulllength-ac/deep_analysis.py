"""Stdlib-only secondary analysis of records.jsonl; writes deep_analysis.json. Usage: python3 deep_analysis.py"""
import json, math, re, statistics as st, collections
from pathlib import Path
here = Path(__file__).parent
rows = [json.loads(l) for l in (here / 'records.jsonl').open()]
old = [json.loads(l) for l in (here.parent / '20261102-fulllength-qa/records.jsonl').open()]
data_rows = {}
for l in (here.parents[2] / 'test_data/longchat.jsonl').open():
    r = json.loads(l); data_rows[r['test_id']] = r
ORDER = ['ac_b16', 'cachegen1', 'ac_b12', 'layer_a', 'ac_b10', 'layer_b', 'layer_c', 'ac_b8', 'layer_d', 'ac_b6']
ARMS = ('author', 'clean_question')
SCHED = {'ac_b16': (16,16,16,16,16), 'ac_b12': (12,)*5, 'ac_b10': (10,)*5, 'ac_b8': (8,)*5, 'ac_b6': (6,)*5,
         'cachegen1': (16,16,12,16,12), 'layer_a': (16,12,10,16,10), 'layer_b': (16,10,8,16,8),
         'layer_c': (12,8,8,12,8), 'layer_d': (12,8,6,12,6)}
get = lambda arm, m: {r['source_id']: r for r in rows if r['arm'] == arm and r['method'] == m}
words = lambda s: ' '.join(re.findall(r'\w+', s.casefold()))
out = {}

# 1. Where the bytes go (author arm). Table/maxima/lengths follow the wire format; the rest is coded symbols.
SEG = 96; ROWS = 64; CH = 1024
sz = {}
for m in ['uniform8_ac'] + ORDER:
    A = get('author', m); t = st.mean(r['kv_tokens'] for r in A.values()); b = st.mean(r['bytes'] for r in A.values())
    if m == 'uniform8_ac':
        table = ROWS*CH*257*2; mode = 'cdf256'
    else:
        table = ROWS*CH*(max(SCHED[m])+1)*2
    maxima = ROWS*t*2
    lengths = math.ceil(t/SEG)*ROWS*CH*4
    lengths256 = math.ceil(t/256)*ROWS*CH*4
    sym = b-table-maxima-lengths
    n = ROWS*t*CH
    sz[m] = dict(mb=b/1e6, table_mb=table/1e6, maxima_mb=maxima/1e6, lengths_mb=lengths/1e6, symbols_mb=sym/1e6,
                 overhead_share=(table+maxima+lengths)/b, bits_per_value=8*sym/n, total_bits_per_value=8*b/n,
                 mb_if_256_token_segments=(b-lengths+lengths256)/1e6)
out['bytes'] = sz

# 2. Document-level failure structure.
fail = {arm: {m: {i for i, r in get(arm, m).items() if not r['includes_answer']} for m in ORDER} for arm in ARMS}
for arm in ARMS:
    fail[arm]['uniform3_old'] = {r['source_id'] for r in old if r['arm'] == arm and r['method'] == 'uniform3' and not r['includes_answer']}
out['fail_sets'] = {arm: {m: sorted(s) for m, s in fail[arm].items()} for arm in ARMS}
nest = {}
for arm in ARMS:
    chain = ['ac_b10', 'ac_b8', 'ac_b6']
    nest[arm] = {f'{a} in {b}': len(fail[arm][a] & fail[arm][b]) / max(1, len(fail[arm][a])) for a, b in zip(chain, chain[1:])}
out['nested'] = nest
hard = collections.Counter()
for arm in ARMS:
    for m in ('ac_b10', 'layer_b', 'layer_c', 'ac_b8'):
        hard.update(fail[arm][m])
out['hard_docs'] = hard.most_common(12)
out['never_failed_mid'] = sorted(set(range(50)) - set(hard))
overlap = {m: dict(author=len(fail['author'][m]), clean=len(fail['clean_question'][m]),
                   both=len(fail['author'][m] & fail['clean_question'][m])) for m in ORDER}
out['arm_overlap'] = overlap

# 3. What wrong answers say: position of the named topic in the 15-topic list.
pos = collections.Counter(); names = collections.Counter(); kinds = collections.Counter(); ptab = {}
for arm in ARMS:
    for m in ('ac_b10', 'layer_b', 'layer_c', 'ac_b8', 'layer_d'):
        ptab[f'{arm}/{m}'] = collections.Counter()
        for i, r in get(arm, m).items():
            if r['includes_answer']: continue
            first = words(r['prediction'].lstrip('\n').split('\n', 1)[0])
            labels = data_rows[i]['label']
            hit = [k for k, t in enumerate(labels) if f' {words(t)} ' in f' {first} ']
            if hit:
                pos[hit[0]] += 1; ptab[f'{arm}/{m}'][hit[0]] += 1; names[labels[hit[0]]] += 1; kinds['other_listed_topic'] += 1
            else:
                kinds['not_a_listed_topic'] += 1
out['wrong_kind'] = dict(kinds); out['wrong_position'] = sorted(pos.items()); out['wrong_topics'] = names.most_common(8)
base = collections.Counter(t for r in data_rows.values() for t in r['label'][1:])
first_freq = collections.Counter(r['label'][0] for r in data_rows.values())
out['topics_in_data'] = dict(distinct=len(base | first_freq), first_topic_distinct=len(first_freq),
                             top_first=first_freq.most_common(6), base_rate_of_top_wrong={k: base[k] for k, _ in names.most_common(5)})
# label sanity: do the labelled topics occur in prompt order?
order_ok = sum(all(data_rows[i]['prompt'].find(a) < data_rows[i]['prompt'].find(b) for a, b in zip(data_rows[i]['label'], data_rows[i]['label'][1:])) for i in data_rows)
out['labels_in_prompt_order'] = order_ok

# 4. More sensitive signals than the ceilinged accuracy.
sig = {}
for arm in ARMS:
    sig[arm] = {}
    for m in ORDER:
        A = get(arm, m)
        sig[arm][m] = dict(acc=sum(r['includes_answer'] for r in A.values()), identical=sum(r['identical_to_raw'] for r in A.values()),
                           mse=st.mean(r['mse'] for r in A.values()),
                           clean_stop=sum(r['stop_reason'] == 'eos' for r in A.values()),
                           empty=sum(len(r['prediction'].strip()) < 5 for r in A.values()))
    A = get(arm, 'raw'); sig[arm]['raw'] = dict(clean_stop=sum(r['stop_reason'] == 'eos' for r in A.values()))
out['signals'] = sig
def spearman(x, y):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0]*len(v); i = 0
        while i < len(v):
            j = i
            while j+1 < len(v) and v[o[j+1]] == v[o[i]]: j += 1
            for k in range(i, j+1): r[o[k]] = (i+j)/2
            i = j+1
        return r
    a, b = rank(x), rank(y); ma, mb = st.mean(a), st.mean(b)
    return sum((p-ma)*(q-mb) for p, q in zip(a, b))/math.sqrt(sum((p-ma)**2 for p in a)*sum((q-mb)**2 for q in b))
for arm in ARMS:
    x = [sig[arm][m]['mse'] for m in ORDER]; y = [sig[arm][m]['acc'] for m in ORDER]
    z = [sz[m]['mb'] for m in ORDER]
    out[f'spearman_{arm}'] = dict(mse_vs_acc=spearman(x, y), size_vs_acc=spearman(z, y))

# 5. Do easy documents explain the id0-19 vs id20-49 gap? Do longer contexts fail more?
gap = {}
for arm in ARMS:
    for m in ('ac_b10', 'layer_c', 'ac_b8', 'layer_d'):
        A = get(arm, m)
        a = sum(A[i]['includes_answer'] for i in range(20)); b = sum(A[i]['includes_answer'] for i in range(20, 50))
        gap[f'{arm}/{m}'] = dict(id0_19=f'{a}/20', id20_49=f'{b}/30')
out['id_gap'] = gap
tok = {i: r['kv_tokens'] for i, r in get('author', 'raw').items()}
s = sorted(tok, key=tok.get); bands = [s[:17], s[17:34], s[34:]]
out['by_length'] = {m: [f"{sum(not get('clean_question', m)[i]['includes_answer'] for i in b)}/{len(b)}" for b in bands] for m in ('ac_b8', 'layer_c', 'layer_d')}
out['length_bands'] = [(tok[b[0]], tok[b[-1]]) for b in bands]

# 6. Layering: same-accuracy size saving and the size-per-accuracy frontier.
front = {}
for arm in ARMS:
    pts = sorted(((sz[m]['mb'], sig[arm][m]['acc'], m) for m in ORDER), reverse=True)
    best = []
    for mb, acc, m in sorted(pts):
        if not best or acc > best[-1][1]:
            best.append((mb, acc, m))
    front[arm] = best
out['pareto'] = front
(here / 'deep_analysis.json').write_text(json.dumps(out, indent=1, ensure_ascii=False, default=list))
print(json.dumps({k: out[k] for k in ('bytes',)}, indent=0)[:0])
for m, v in sz.items(): print(f"{m:12s} MB={v['mb']:.1f} sym={v['symbols_mb']:.1f} table={v['table_mb']:.1f} max={v['maxima_mb']:.1f} len={v['lengths_mb']:.1f} over={v['overhead_share']:.1%} bits={v['bits_per_value']:.2f} total_bits={v['total_bits_per_value']:.2f} if256={v['mb_if_256_token_segments']:.1f}")
for k in ('fail_sets', 'nested', 'hard_docs', 'never_failed_mid', 'arm_overlap', 'wrong_kind', 'wrong_position', 'wrong_topics', 'topics_in_data', 'labels_in_prompt_order', 'signals', 'spearman_author', 'spearman_clean_question', 'id_gap', 'by_length', 'length_bands', 'pareto'):
    print(k, json.dumps(out[k], ensure_ascii=False, default=list)[:1500])

# 7. Fidelity signals below the accuracy ceiling (paired exact tests, no multiplicity correction).
def mcnemar(b, c):
    n = b+c
    return 1.0 if not n else min(1.0, 2*sum(math.comb(n, i) for i in range(min(b, c)+1))/2**n)
fid = {}
for arm in ARMS:
    for a, b in (('layer_a','ac_b12'),('layer_a','ac_b10'),('cachegen1','ac_b16'),('cachegen1','ac_b12'),('ac_b12','ac_b16'),('layer_b','ac_b10'),('layer_b','ac_b8'),('layer_c','ac_b8')):
        A, B = get(arm, a), get(arm, b)
        row = {}
        for key, f in (('identical', lambda r: r['identical_to_raw']), ('eos', lambda r: r['stop_reason'] == 'eos')):
            x = sum(f(A[i]) and not f(B[i]) for i in A); y = sum(f(B[i]) and not f(A[i]) for i in A)
            row[key] = dict(a=sum(f(A[i]) for i in A), b=sum(f(B[i]) for i in A), a_only=x, b_only=y, p=mcnemar(x, y))
        fid[f'{arm}: {a} vs {b}'] = row
out['fidelity'] = fid

# 8. Topic order: use the real positions of the transition phrase, not the label list order.
order = {}
real_pos = collections.Counter(); first_ok = 0; seq_ok = 0; found_all = 0
for i, d in data_rows.items():
    p = d['prompt']; found = [p.find(t) for t in d['label']]
    found_all += all(x >= 0 for x in found)
    ok = [x for x in found if x >= 0]
    first_ok += bool(found[0] >= 0 and found[0] == min(ok))
    seq_ok += ok == sorted(ok)
out['label_check'] = dict(docs=len(data_rows), all_labels_found=found_all, label0_is_earliest=first_ok, labels_sorted_by_first_occurrence=seq_ok)
ex = data_rows[0]; out['label_example'] = [(t, ex['prompt'].find(t)) for t in ex['label'][:15]]
real = collections.Counter(); notfound = 0
for arm in ARMS:
    for m in ('ac_b10', 'layer_b', 'layer_c', 'ac_b8', 'layer_d'):
        for i, r in get(arm, m).items():
            if r['includes_answer']: continue
            first = words(r['prediction'].lstrip('\n').split('\n', 1)[0]); d = data_rows[i]
            named = [t for t in d['label'][1:] if f' {words(t)} ' in f' {first} ']
            if not named: notfound += 1; continue
            spots = sorted(d['prompt'].find(t) for t in d['label'] if d['prompt'].find(t) >= 0)
            at = d['prompt'].find(named[0])
            real[spots.index(at) if at in spots else -1] += 1
out['wrong_real_position'] = sorted(real.items()); out['wrong_unlisted'] = notfound
(here / 'deep_analysis.json').write_text(json.dumps(out, indent=1, ensure_ascii=False, default=list))
print('--- fidelity')
for k, v in fid.items(): print(k, 'identical', v['identical'], 'eos', v['eos'])
print(out['label_check']); print(out['label_example'][:6]); print('real', out['wrong_real_position'], out['wrong_unlisted'])
print(repr(ex['prompt'][240:760]))
