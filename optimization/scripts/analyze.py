"""Aggregate recorded paired observations without refitting frozen profiles."""
import argparse
from collections import defaultdict
import csv
import html
import re
from pathlib import Path
import statistics
import sys

repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo))
from experiments.joint_config import run_directory
from experiments.joint_records import benchmark_directory, read_benchmark_records

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run-dir', required=True)
parser.add_argument('--diagnostic-only', action='store_true', help='analyze separate diagnostic observations')
args = parser.parse_args()
run = benchmark_directory(run_directory(args.run_dir), args.diagnostic_only)
records = read_benchmark_records(run / 'records.jsonl', args.diagnostic_only)
if not records:
    parser.error('No completed benchmark observations in the selected scope')
groups = defaultdict(list)
for row in records:
    groups[(row['group'], row['T'], row['execution_mode'], row['target_gbps'], row['slo_s'],
            row['cache_condition'], row.get('trajectory','stable'))].append(row)
fields = ['group','T','mode','target_gbps','slo_s','cache_condition','trajectory',
          'requests','sources','ttft_median_s','ttft_p90_s','slo_fraction','generated_packets','wire_bytes']
with (run / 'summary.csv').open('w', newline='', encoding='utf-8') as stream:
    writer = csv.writer(stream)
    writer.writerow(fields)
    for key, rows in sorted(groups.items()):
        values = sorted(r['ttft_s'] for r in rows)
        n = len(values)
        wire_bytes = sum(r.get('wire_bytes', sum(e.get('wire_bytes',0) for e in r.get('network',[]))) for r in rows)
        writer.writerow([*key,n,len({r['source_id'] for r in rows}),statistics.median(values),
                         values[max(0, (9*n+9)//10-1)],sum(r['slo_met'] for r in rows)/n,
                         sum(len(r.get('generation',[])) for r in rows),wire_bytes])
scope = 'diagnostic' if args.diagnostic_only else 'measured'
text = [f'# Joint-v2 {scope} experiment report', '',
        f'Recorded requests: {len(records)}. All values derive from completed observations.', '',
        '[Grouped metrics](summary.csv) · [Raw observations](records.jsonl)', '',
        'P50 is the median. P90 uses the nearest-rank statistic. The reported SLO fraction uses actual TTFT.', '',
        'Source-level pairing preserves source, T, mode, bandwidth, SLO, cache condition, trajectory and repetition.']
paired = defaultdict(list)
index = {}
for row in records:
    condition = tuple(row[k] for k in ('source_id','T','execution_mode','target_gbps','slo_s','cache_condition','repeat')) + (row.get('trajectory','stable'),)
    index[(condition,row['group'])] = row
for row in records:
    match = re.search(r'-q([123])$',row['group'])
    control = 'original-q'+match.group(1) if match else 'original-adaptive-history'
    if row['group'] == control:
        continue
    condition = tuple(row[k] for k in ('source_id','T','execution_mode','target_gbps','slo_s','cache_condition','repeat')) + (row.get('trajectory','stable'),)
    baseline = index.get((condition,control))
    if baseline is not None:
        paired[(row['group'],control,row['T'],row['cache_condition'])].append(row['ttft_s']-baseline['ttft_s'])
with (run/'paired.csv').open('w',newline='',encoding='utf-8') as stream:
    writer=csv.writer(stream)
    writer.writerow(['group','control','T','cache_condition','paired_requests','median_delta_s'])
    for key,values in sorted(paired.items()):
        writer.writerow([*key,len(values),statistics.median(values)])
text.extend(['','[Paired TTFT differences](paired.csv). Negative differences indicate shorter observed TTFT for the treatment.'])
bars=[]
for key,rows in sorted(groups.items()):
    group,total,mode,bw,slo,cache,trace=key
    if total == 4096 and mode == 'pipeline' and bw == 1. and slo == 1. and cache == 'cold' and trace == 'stable':
        bars.append((group,statistics.median(r['ttft_s'] for r in rows)))
if bars:
    maximum=max(v for _,v in bars) or 1.
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="{65+len(bars)*32}" viewBox="0 0 900 {65+len(bars)*32}">',
         '<rect width="100%" height="100%" fill="white"/>',
         '<text x="20" y="26" font-family="sans-serif" font-size="16">4K / pipeline / cold / target 1 Gbps / SLO 1s — observed median TTFT</text>']
    for i,(name,value) in enumerate(bars):
        y=45+32*i
        svg.extend([f'<text x="20" y="{y+17}" font-family="sans-serif" font-size="12">{html.escape(name)}</text>',
                    f'<rect x="245" y="{y}" width="{550*value/maximum:.2f}" height="23" fill="#3568a8"/>',
                    f'<text x="{255+550*value/maximum:.2f}" y="{y+17}" font-family="sans-serif" font-size="12">{value:.4f}s</text>'])
    svg.append('</svg>')
    (run/'ttft-4k-pipeline-cold.svg').write_text('\n'.join(svg),encoding='utf-8')
    text.extend(['','![Observed medians](ttft-4k-pipeline-cold.svg)'])
(run / 'report.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
print(run / 'report.md')
