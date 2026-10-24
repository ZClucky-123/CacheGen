"""Wait for the detached experiment, then build and verify its artifacts."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def save(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('run_dir',type=Path);args=p.parse_args()
    run=args.run_dir;root=Path(__file__).resolve().parent
    source=run/'source';source.mkdir(exist_ok=True)
    for path in list(root.glob('*.py'))+[root/'run.sh',root/'config.json',root/'README.md']:
        shutil.copy2(path,source/path.name)
    while not (run/'exit_code').exists():time.sleep(5)
    code=int((run/'exit_code').read_text())
    if code:
        save(run/'artifacts.json',{'status':'skipped','reason':'measurement process failed','exit_code':code});return code
    env=os.environ.copy();env['PYTHONPATH']=str(root/'plotdeps');env['MPLCONFIGDIR']=str(root/'mplconfig')
    subprocess.run(['/opt/conda/bin/python',str(root/'plot_results.py'),str(run)],env=env,check=True)
    summary=json.loads((run/'summary.json').read_text());manifest=json.loads((run/'manifest.json').read_text())
    if manifest['status']!='completed':raise RuntimeError('Incomplete measurement batch')
    meta=json.loads((run/'metadata.json').read_text())
    mismatch=[path for path,expected in meta['source_and_data_sha256'].items() if digest(Path(path))!=expected]
    if mismatch:raise RuntimeError(f'Source/data changed during measurements: {mismatch}')
    quality=summary['quality'];qa={r['method']:r for r in summary['qa']};timing=summary['ttft']
    cg3=[r for r in quality if r['method']=='cachegen3']
    cg3_ttft=[r for r in timing if r['method']=='cachegen3']
    conclusion=['## 本批次结果解释',
        'CacheGen 3 的“原始序列化 KV / 压缩 KV”体积比分别为 '+
        '、'.join(f"{r['context_tokens']} token：{r['raw_serialized_over_method']:.3f}×" for r in cg3)+'。',
        '对应配对 NLL 增量分别为 '+ '、'.join(f"{r['paired_nll_delta']:+.5f}" for r in cg3)+'。',
        f"10 条完整 TriviaQA 上，原始 KV 的 EM/F1 为 {qa['raw']['mean_em']:.3f}/{qa['raw']['mean_f1']:.3f}；CacheGen 3 为 {qa['cachegen3']['mean_em']:.3f}/{qa['cachegen3']['mean_f1']:.3f}。",
        f"9 个长度/带宽组合中，CacheGen 3 有 {sum(r['speedup_vs_text']>1 for r in cg3_ttft)}/9 个组合的中位 TTFT 优于文本重算，{sum(r['speedup_vs_uniform8']>1 for r in cg3_ttft)}/9 个组合优于 uniform8。应结合上面的质量差值判断适用条件。",
        '不同压缩档位的质量和延迟取舍以本批次实测为准。单卡、短上下文、本机限速网络结果不能外推为论文原模型规模下的完整系统收益。',
        '## 图表',
        '![压缩体积与续写质量](figures/compression_quality.png)',
        '![完整 TriviaQA](figures/qa.png)',
        '![首 token 延迟](figures/ttft.png)',
        'figures/ 同时提供可缩放的 SVG。source/ 保存本次实验入口、配置和后处理代码快照。']
    if qa['raw']['raw_em_correct']==0:
        conclusion.insert(1,
            '本批次原始 KV 的完整输出 EM 正确题数为 0，因此“原始正确题保留率”不可计算，预设的联合质量门槛未获支持。'
            'quality.csv 中的 False 表示没有足够证据通过该门槛，不表示所有压缩方法都比原始 KV 差。'
            '本轮按事先固定的完整生成输出计算 EM/F1；模型输出附加内容也会降低严格 EM，故需结合逐题预测和配对 NLL 阅读。')
    with (run/'report.md').open('a') as stream:stream.write('\n\n'+'\n\n'.join(conclusion)+'\n')
    files=['quality.json','ttft.json','smoke.json','summary.json','quality.csv','qa.csv','ttft.csv','report.md',
           'figures/compression_quality.png','figures/compression_quality.svg','figures/ttft.png','figures/ttft.svg','figures/qa.png','figures/qa.svg']
    records={name:{'bytes':(run/name).stat().st_size,'sha256':digest(run/name)} for name in files}
    if any(row['bytes']==0 for row in records.values()):raise RuntimeError('Empty artifact')
    save(run/'artifacts.json',{'status':'completed','finished_utc':datetime.now(timezone.utc).isoformat(),
                              'source_data_fingerprints_match':True,'files':records,
                              'plot_python':'/opt/conda/bin/python','plot_script_sha256':digest(root/'plot_results.py')})
    print('ARTIFACTS_COMPLETED',run,flush=True)
    return 0


if __name__=='__main__':
    try:sys.exit(main())
    except BaseException as error:
        if not isinstance(error,SystemExit):
            for i,arg in enumerate(sys.argv):
                if i==1:save(Path(arg)/'artifacts.json',{'status':'failed','error':repr(error)})
        raise
