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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.workspace import Workspace


def save(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_source_fingerprints(run, repo):
    meta=json.loads((run/'metadata.json').read_text())
    relative=meta.get('source_and_data_relative_sha256', {})
    mismatch=[]
    for name,expected in meta['source_and_data_sha256'].items():
        recorded=Path(name)
        snapshot=run/'source'/recorded.name
        if name==meta.get('configuration_file'):snapshot=run/'source/config.json'
        candidates=[snapshot,recorded]
        candidates += [repo/path for path,value in relative.items() if value==expected]
        if not any(path.is_file() and digest(path)==expected for path in candidates):mismatch.append(name)
    if mismatch:raise RuntimeError(f'Source/data fingerprints could not be verified: {mismatch}')


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('run_dir',type=Path);args=p.parse_args(argv)
    run=args.run_dir;root=Path(__file__).resolve().parent
    workspace=Workspace.discover()
    manifest_path=run/'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()).get('protocol') in ('cachegen-shared-v2', 'cachegen-shared-v3'):
        from experiments.artifacts import verify
        if json.loads(manifest_path.read_text())['status']!='completed':
            raise RuntimeError('Shared batch is incomplete; resume it with experiments/run.sh --resume')
        print('SHARED_ARTIFACTS_VERIFIED', verify(run))
        return 0
    if (run/'artifacts.json').is_file():
        archived=json.loads((run/'artifacts.json').read_text())
        if archived.get('status')=='completed':
            for name,row in archived['files'].items():
                path=run/name
                if path.stat().st_size!=row['bytes'] or digest(path)!=row['sha256']:
                    raise RuntimeError(f'Archived artifact changed: {name}')
            print('ARTIFACTS_ALREADY_COMPLETED',run)
            return 0
    source=run/'source';source.mkdir(exist_ok=True)
    for path in list(root.glob('*.py'))+[root/'run.sh',root/'config.json',root/'README.md']:
        if not (source/path.name).exists():shutil.copy2(path,source/path.name)
    while not (run/'exit_code').exists():time.sleep(5)
    code=int((run/'exit_code').read_text())
    if code:
        save(run/'artifacts.json',{'status':'skipped','reason':'measurement process failed','exit_code':code});return code
    env=os.environ.copy()
    deps=workspace.local/'mistral/plotdeps'
    if deps.is_dir():env['PYTHONPATH']=str(deps)+os.pathsep+env.get('PYTHONPATH','')
    env['MPLCONFIGDIR']=str(workspace.local/'mistral/mplconfig')
    plot_python=os.environ.get('CACHEGEN_PLOT_PYTHON',sys.executable)
    subprocess.run([plot_python,str(root/'plot_results.py'),str(run)],env=env,check=True)
    summary=json.loads((run/'summary.json').read_text());manifest=json.loads((run/'manifest.json').read_text())
    if manifest['status']!='completed':raise RuntimeError('Incomplete measurement batch')
    meta=json.loads((run/'metadata.json').read_text())
    verify_source_fingerprints(run,workspace.repo)
    quality=summary['quality'];qa={r['method']:r for r in summary['qa']};timing=summary['ttft']
    cg3=[r for r in quality if r['method']=='cachegen3']
    cg3_ttft=[r for r in timing if r['method']=='cachegen3']
    conclusion=['## 本批次结果解释',
        'CacheGen 3 的“原始序列化 KV / 压缩 KV”体积比分别为 '+
        '、'.join(f"{r['context_tokens']} token：{r['raw_serialized_over_method']:.3f}×" for r in cg3)+'。',
        '对应配对 NLL 增量分别为 '+ '、'.join(f"{r['paired_nll_delta']:+.5f}" for r in cg3)+'。',
        f"{qa['raw']['questions']} 条完整 TriviaQA 上，原始 KV 的 EM/F1 为 {qa['raw']['mean_em']:.3f}/{qa['raw']['mean_f1']:.3f}；CacheGen 3 为 {qa['cachegen3']['mean_em']:.3f}/{qa['cachegen3']['mean_f1']:.3f}。",
        f"{len(cg3_ttft)} 个长度/带宽组合中，CacheGen 3 有 {sum(r['speedup_vs_text']>1 for r in cg3_ttft)}/{len(cg3_ttft)} 个组合的中位 TTFT 优于文本重算，{sum(r['speedup_vs_uniform8']>1 for r in cg3_ttft)}/{len(cg3_ttft)} 个组合优于 uniform8。应结合上面的质量差值判断适用条件。",
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
                              'plot_python':plot_python,'plot_script_sha256':digest(root/'plot_results.py')})
    print('ARTIFACTS_COMPLETED',run,flush=True)
    return 0


if __name__=='__main__':
    try:sys.exit(main())
    except BaseException as error:
        if not isinstance(error,SystemExit):
            for i,arg in enumerate(sys.argv):
                if i==1:
                    destination=Path(arg)/'artifacts.json'
                    if destination.exists() and json.loads(destination.read_text()).get('status')=='completed':
                        destination=Path(arg)/'artifact-verification-error.json'
                    save(destination,{'status':'failed','error':repr(error)})
        raise
