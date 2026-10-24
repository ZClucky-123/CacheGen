import hashlib,json,shutil,sys
from pathlib import Path
repo=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(repo))
from experiments.artifacts import verify,sha256
run=repo/'results/mistral/small/v2-small-final-20261024'
assert json.loads((run/'manifest.json').read_text())['status']=='completed'
verify(run)
out=repo/'reports/mistral/v2-small-final-20261024'
out.mkdir(parents=True,exist_ok=False)
text=(repo/'docs/reproduction/SHARED_FRAMEWORK_RESULTS_20261024.md').read_text()
for name in ['shared-helper-validation-20261024.json','SHARED_FRAMEWORK_AUDIT.md','HISTORY.md','larger-mistral-preflight-20261024.json','larger-tinyllama-preflight-20261024.json','SHARED_FRAMEWORK.md']:
    text=text.replace(']('+name+')','](../../../docs/reproduction/'+name+')')
text=text.replace('../../results/','../../../results/')
text=text.replace('../../../results/mistral/small/v2-small-final-20261024/figures/','figures/')
text+='\n\n逐例生成与固定评分见 [问答逐例对照](问答逐例对照.md)。本中文说明对应当前验收批次；较大配置自动生成的 report.md/CSV/JSON/图表由公共 analysis 模块负责。\n'
(out/'验收报告.md').write_text(text)
shutil.copytree(run/'figures',out/'figures')
shutil.copy2(run/'config.json',out/'config.json')
shutil.copy2(run/'summary.json',out/'summary.json')
(out/'code').mkdir()
for name in ['build_shared_acceptance.py','package_shared_report.py']:
    shutil.copy2(repo/'.local'/name,out/'code'/name)
rows=[json.loads(line) for line in (run/'records.jsonl').open()]
groups={}
for row in rows:
    if row['kind'] in ('controlled','triviaqa'):
        groups.setdefault((row['kind'],row['case_id']),[]).append(row)
qa=['# 问答逐例对照','所有分数直接引用主批次，未重评分。scored_answer 是固定首行截取后的文本；下面的生成保留完整内容。原始 token IDs 见 records.jsonl。受控短语包含只作诊断，不代替 EM/F1。']
for (kind,case_id),values in sorted(groups.items()):
    row=values[0]
    qa += [f'## {kind} / {case_id}',f'上下文 token：{row["context_tokens"]}；位置：{row.get("position","独立真实 QA")}；答案别名：{json.dumps(row["answers"],ensure_ascii=False)}']
    if 'question' in row:qa.append(row['question'])
    for value in values:
        qa += [f'**{value["method"]}**：EM={value["em"]:.3f}，F1={value["f1"]:.3f}，stop={value["stop_reason"]}；scored_answer={json.dumps(value["scored_answer"],ensure_ascii=False)}',
               '````text\n'+value['prediction']+'\n````']
(out/'问答逐例对照.md').write_text('\n\n'.join(qa)+'\n')
(out/'source-reference.json').write_text(json.dumps({'run_directory':'../../../results/mistral/small/v2-small-final-20261024','artifacts_sha256':sha256(run/'artifacts.json'),'source_directory':'../../../results/mistral/small/v2-small-final-20261024/source','source_fingerprint_evidence':'../../../docs/reproduction/shared-helper-validation-20261024.json'},indent=2)+'\n')
files={str(p.relative_to(out)):{'bytes':p.stat().st_size,'sha256':sha256(p)} for p in sorted(out.rglob('*')) if p.is_file()}
(out/'report_manifest.json').write_text(json.dumps({'status':'completed','primary_run':'results/mistral/small/v2-small-final-20261024','files':files},indent=2)+'\n')
nav='# 公共框架小规模验收导航\n\n[完整中文验收报告](../../reports/mistral/v2-small-final-20261024/验收报告.md) · [问答逐例对照](../../reports/mistral/v2-small-final-20261024/问答逐例对照.md) · [原始批次](../../results/mistral/small/v2-small-final-20261024/)\n\nMistral：6/6 任务、552/552 正式观测，退出 0。TinyLlama：相同源码的最小 6/6 任务、134/134 观测。框架完成不等于所有质量门槛通过，负面结果见报告。\n\n[运行、后台、恢复和较大配置](SHARED_FRAMEWORK.md) · [初始审查](SHARED_FRAMEWORK_AUDIT.md) · [历史索引](HISTORY.md) · [相同源码证据](shared-helper-validation-20261024.json)\n'
(repo/'docs/reproduction/SHARED_FRAMEWORK_RESULTS_20261024.md').write_text(nav)
print(out)
