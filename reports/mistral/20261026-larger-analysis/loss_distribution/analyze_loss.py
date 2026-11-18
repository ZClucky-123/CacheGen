"""Reproduce paired natural-continuation loss diagnostics without GPU/model loading."""
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics as st

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
RUN = ROOT / 'results/mistral/larger/v3-larger-stable-20261025'
METHODS = ['uniform4', 'uniform8', 'cachegen1', 'cachegen2', 'cachegen3']


def csv_file(name, rows):
    with (OUT / name).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def correlation(xs, ys):
    xm, ym = st.mean(xs), st.mean(ys)
    den = math.sqrt(sum((x-xm)**2 for x in xs) * sum((y-ym)**2 for y in ys))
    return sum((x-xm)*(y-ym) for x, y in zip(xs, ys)) / den if den else None


def main():
    manifest = json.loads((RUN / 'artifacts.json').read_text())
    for name in ('records.jsonl', 'selection.json', 'summary.json'):
        assert hashlib.sha256((RUN/name).read_bytes()).hexdigest() == manifest['files'][name]['sha256']
    rows = []
    with (RUN/'records.jsonl').open() as stream:
        for line in stream:
            r = json.loads(line)
            if r['kind'] == 'quality':
                rows.append(r)
    assert len(rows) == 360
    raw = {(r['source_id'], r['context_tokens']):r for r in rows if r['method']=='raw'}
    paired, tokens = [], []
    for r in rows:
        b = raw[(r['source_id'], r['context_tokens'])]
        assert r['scored_ids'] == b['scored_ids']
        assert len(r['token_nll']) == r['tokens_scored'] == 32
        assert abs(st.mean(r['token_nll']) - r['nll']) < 1e-6
        if r['method']=='raw':
            continue
        delta = r['nll']-b['nll']
        paired.append({'source_id':r['source_id'], 'context_tokens':r['context_tokens'],
                       'method':r['method'], 'raw_nll':b['nll'], 'compressed_nll':r['nll'],
                       'delta_nll':delta, 'kv_mae':r['kv_mae'], 'bytes':r['bytes']})
        for i, (a, z, token) in enumerate(zip(r['token_nll'], b['token_nll'], r['scored_ids']), 1):
            tokens.append({'source_id':r['source_id'], 'context_tokens':r['context_tokens'],
                           'method':r['method'], 'continuation_position':i, 'token_id':token,
                           'raw_token_nll':z, 'compressed_token_nll':a, 'delta_token_nll':a-z})
    assert len(paired)==300 and len(tokens)==9600
    archived=json.loads((RUN/'summary.json').read_text())
    summaries=[]
    for length in (1024, 2048, 4096):
        for method in METHODS:
            q=[r for r in paired if r['context_tokens']==length and r['method']==method]
            ds=sorted(r['delta_nll'] for r in q)
            ref=next(r for r in archived['quality'] if r['context_tokens']==length and r['method']==method)
            assert len(ds)==20 and abs(st.mean(ds)-ref['paired_nll_delta'])<1e-12
            summaries.append({'context_tokens':length,'method':method,'sources':20,
                'mean_delta':st.mean(ds),'median_delta':st.median(ds),'min_delta':min(ds),'max_delta':max(ds),
                'positive_sources':sum(d>0 for d in ds),'sources_above_0_02':sum(d>.02 for d in ds),
                'mean_kv_mae':st.mean(r['kv_mae'] for r in q),
                'mae_loss_pearson_descriptive':correlation([r['kv_mae'] for r in q],[r['delta_nll'] for r in q]),
                'mean_delta_cluster_ci95':ref['paired_nll_cluster_ci95']})
    target=sorted((r for r in paired if r['context_tokens']==4096 and r['method']=='cachegen1'),
                  key=lambda r:r['delta_nll'],reverse=True)
    ds=[r['delta_nll'] for r in target]
    sensitivity={'exclude_largest_mean':st.mean(ds[1:]),'exclude_largest_3_mean':st.mean(ds[3:]),
                 'leave_one_out_mean_range':[min((sum(ds)-d)/19 for d in ds),max((sum(ds)-d)/19 for d in ds)],
                 'note':'Sensitivity only. No observation is removed from the formal result.'}
    tt=[r for r in tokens if r['context_tokens']==4096 and r['method']=='cachegen1']
    positives=sorted((r['delta_token_nll'] for r in tt if r['delta_token_nll']>0),reverse=True)
    evidence={'source_batch':str(RUN.relative_to(ROOT)),'paired_document_rows':len(paired),'paired_token_rows':len(tokens),
              'distribution':summaries,'sensitivity_4k_cachegen1':sensitivity,
              'token_4k_cachegen1':{'count':len(tt),'positive_tokens':len(positives),
                  'top10_share_of_positive_loss_sum':sum(positives[:10])/sum(positives)},
              'hypothesis':'Coarse quantization and content-specific probability sensitivity; layer/K/V causes unverified.'}
    (OUT/'evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
    csv_file('document_deltas.csv',paired)
    csv_file('token_deltas.csv',tokens)
    csv_file('ranked_4k_cachegen1.csv',target)
    make_plots(paired,tokens,target)
    print(json.dumps({'document_pairs':len(paired),'token_pairs':len(tokens),'sensitivity':sensitivity},ensure_ascii=False))


def make_plots(paired,tokens,target):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import numpy as np
    colors={'cachegen1':'#c66a35','cachegen2':'#287e90','cachegen3':'#785c92'}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    ids=[r['source_id'] for r in target]
    with PdfPages(OUT/'loss-distribution.pdf') as pdf:
        fig,axes=plt.subplots(2,1,figsize=(12,8),layout='constrained')
        for j,(method,color) in enumerate(colors.items()):
            vals={r['source_id']:r['delta_nll'] for r in paired if r['context_tokens']==4096 and r['method']==method}
            axes[0].bar(np.arange(20)+(j-1)*.24,[vals[i] for i in ids],width=.24,color=color,label=method)
        axes[0].axhline(.02,color='#a73030',ls='--',label='Batch mean gate (not per-document gate)')
        axes[0].axhline(0,color='gray',lw=.6)
        axes[0].set(xticks=np.arange(20),xticklabels=ids,xlabel='Source ID (sorted by 4K CacheGen1 loss)',
                    ylabel='Paired NLL change',title='4K: loss varies substantially by source; each bar averages 32 target tokens')
        axes[0].legend(fontsize=8,ncol=2)
        for length in (1024,2048,4096):
            ds=sorted(r['delta_nll'] for r in paired if r['context_tokens']==length and r['method']=='cachegen1')
            axes[1].step(ds,np.arange(1,21)/20,where='post',label=f'{length} tokens',lw=2)
        axes[1].axvline(.02,color='#a73030',ls='--')
        axes[1].set(xlabel='Paired NLL change (CacheGen1)',ylabel='Fraction of source documents',
                    title='Distribution across 20 sources; different lengths score different continuation windows',ylim=(0,1.05))
        axes[1].legend();axes[1].grid(alpha=.2)
        fig.savefig(OUT/'document-distribution.png',dpi=180);pdf.savefig(fig);plt.close(fig)

        fig,axes=plt.subplots(2,1,figsize=(12,8),layout='constrained')
        lookup={(r['source_id'],r['continuation_position']):r['delta_token_nll'] for r in tokens
                if r['context_tokens']==4096 and r['method']=='cachegen1'}
        heat=np.array([[lookup[(i,k)] for k in range(1,33)] for i in ids])
        limit=float(np.abs(heat).max())
        im=axes[0].imshow(heat,aspect='auto',cmap='RdBu_r',vmin=-limit,vmax=limit)
        axes[0].set(yticks=range(20),yticklabels=ids,xticks=[0,7,15,23,31],xticklabels=[1,8,16,24,32],
                    xlabel='Teacher-forced continuation token position',ylabel='Source ID',
                    title='4K CacheGen1: individual token loss changes (red worse, blue better)')
        fig.colorbar(im,ax=axes[0],label='Token NLL change')
        for method,color in colors.items():
            q=[r for r in paired if r['context_tokens']==4096 and r['method']==method]
            axes[1].scatter([r['kv_mae'] for r in q],[r['delta_nll'] for r in q],color=color,label=method)
        axes[1].axhline(.02,color='#a73030',ls='--')
        axes[1].set(xlabel='Mean absolute KV reconstruction error',ylabel='Paired document NLL change',
                    title='Precision changes error magnitude, but global KV MAE alone does not explain source sensitivity')
        axes[1].legend();axes[1].grid(alpha=.2)
        fig.savefig(OUT/'token-sensitivity.png',dpi=180);pdf.savefig(fig);plt.close(fig)


if __name__=='__main__':
    main()
