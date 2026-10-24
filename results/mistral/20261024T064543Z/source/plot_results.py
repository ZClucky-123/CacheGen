"""Create standalone PNG/SVG figures from a completed measurement batch."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('run_dir',type=Path);args=p.parse_args()
    root=args.run_dir
    manifest=json.loads((root/'manifest.json').read_text())
    if manifest['status']!='completed':raise RuntimeError('Measurements must be completed first')
    data=json.loads((root/'summary.json').read_text())
    out=root/'figures';out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    methods=['raw','uniform4','uniform8','cachegen1','cachegen2','cachegen3']
    labels=['FP16 KV','Uniform 4','Uniform 8','CacheGen 1','CacheGen 2','CacheGen 3']
    colors=dict(zip(methods,['#777777','#d99b38','#c26023','#8d78ad','#3c96a2','#216348']))
    lengths=sorted({r['context_tokens'] for r in data['quality']})
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    x=np.arange(len(lengths));width=.14
    for i,method in enumerate(methods[1:]):
        rows=[next(r for r in data['quality'] if r['method']==method and r['context_tokens']==length) for length in lengths]
        axes[0].bar(x+(i-2)*width,[r['raw_serialized_over_method'] for r in rows],width,color=colors[method],label=labels[i+1])
        axes[1].plot(lengths,[r['paired_nll_delta'] for r in rows],marker='o',color=colors[method],label=labels[i+1])
    axes[0].set(xticks=x,xticklabels=lengths,xlabel='Context tokens',ylabel='FP16 serialized bytes / method bytes',title='Compression including metadata')
    axes[0].axhline(1,color='black',linewidth=.7)
    axes[1].set(xlabel='Context tokens',ylabel='Paired NLL increase',title='Teacher-forced continuation quality')
    axes[1].axhline(.02,color='#333333',linestyle='--',label='Descriptive threshold 0.02')
    axes[1].axhline(0,color='#888888',linewidth=.7)
    axes[0].legend(fontsize=8);axes[1].legend(fontsize=8)
    fig.suptitle('Mistral-7B / RTX 4090 D / shared 8-bit weights / FP16 KV',fontsize=12)
    for suffix in ['png','svg']:fig.savefig(out/f'compression_quality.{suffix}',dpi=180)
    plt.close(fig)
    bandwidths=sorted({r['bandwidth_gbps'] for r in data['ttft']})
    timing_methods=['text','uniform8','cachegen2','cachegen3']
    timing_labels=['Text','Uniform 8','CacheGen 2','CacheGen 3']
    fig,axes=plt.subplots(len(lengths),len(bandwidths),figsize=(13,10),layout='constrained',squeeze=False)
    for i,length in enumerate(lengths):
        for j,bw in enumerate(bandwidths):
            ax=axes[i,j]
            rows=[next(r for r in data['ttft'] if r['context_tokens']==length and r['bandwidth_gbps']==bw and r['method']==m) for m in timing_methods]
            values=[r['median_seconds']*1000 for r in rows]
            high=[max(0,(r['p95_seconds']-r['median_seconds'])*1000) for r in rows]
            ax.bar(range(4),values,color=['#777777',colors['uniform8'],colors['cachegen2'],colors['cachegen3']],
                   yerr=np.array([[0]*4,high]),capsize=3)
            ax.set(xticks=range(4),xticklabels=timing_labels,title=f'{length} tokens / {bw:g} Gbps',ylabel='TTFT (ms)')
            ax.tick_params(axis='x',labelrotation=20)
            ax.grid(axis='y',alpha=.18)
    fig.suptitle('Local paced socket: median TTFT; upper whisker = pooled P95\n3 contexts x 5 measurements; offline encoding excluded',fontsize=12)
    for suffix in ['png','svg']:fig.savefig(out/f'ttft.{suffix}',dpi=180)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4.5),layout='constrained')
    qa={r['method']:r for r in data['qa']};x=np.arange(len(methods))
    ax.bar(x-.17,[qa[m]['mean_em'] for m in methods],.34,label='Exact match',color='#327366')
    ax.bar(x+.17,[qa[m]['mean_f1'] for m in methods],.34,label='Token F1',color='#d99b38')
    ax.set(xticks=x,xticklabels=labels,ylim=(0,1.1),ylabel='Score',title='10 complete TriviaQA inputs (<=4096 tokens), 32-token generation')
    ax.legend();ax.grid(axis='y',alpha=.18)
    for suffix in ['png','svg']:fig.savefig(out/f'qa.{suffix}',dpi=180)
    plt.close(fig)
    print(json.dumps({'figures':str(out),'files':sorted(p.name for p in out.iterdir())}))


if __name__=='__main__':main()
