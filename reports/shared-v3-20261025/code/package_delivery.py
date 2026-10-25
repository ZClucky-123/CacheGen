"""Package verification evidence; experiment metrics come only from sealed runs."""
import hashlib,json,shutil,sqlite3,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from experiments.artifacts import verify,sha256
from experiments.records import save

root=Path.cwd()
runs={n:root/f'results/{n}/minimal/v3-final-03' for n in ('tinyllama','mistral')}
meta={n:json.loads((p/'manifest.json').read_text()) for n,p in runs.items()}
assert all(m['status']=='completed' and m['exit_code']==0 and m['expected_observations']==274 for m in meta.values())
assert meta['tinyllama']['identity']['inputs']['code']==meta['mistral']['identity']['inputs']['code']
for p in runs.values(): verify(p)
prior=json.loads(Path('.local/v3-recovery-before-03.json').read_text())
db=sqlite3.connect('file:'+str(runs['mistral']/'records.sqlite3')+'?mode=ro',uri=True)
current=dict(db.execute('select key,sha256 from observations'));db.close()
assert all(current.get(k)==v for k,v in prior['prior_observations'].items())
records={n:[json.loads(line) for line in (p/'records.jsonl').open(encoding='utf-8')] for n,p in runs.items()}
pairs={}
for r in records['mistral']:
 if r['kind']=='ttft' and r['mode'] in ('chunk_serial','chunk_pipeline'):
  key=tuple(r[k] for k in ('case_id','method','bandwidth_gbps','chunk_size','repetition'))
  pairs.setdefault(key,{})[r['mode']]=r
cross=0
for modes in pairs.values():
 a,b=modes['chunk_serial'],modes['chunk_pipeline']
 assert a['prepared_payload_sha256']==b['prepared_payload_sha256']
 assert a['first_token_id']==b['first_token_id']
 cross+=a['attempt']!=b['attempt']
assert cross>0
proof={'protocol':'cachegen-shared-v3','same_measured_source_map':True,
 'measured_source_files':len(meta['tinyllama']['identity']['inputs']['code']),
 'source_sha256':meta['tinyllama']['identity']['inputs']['code'],
 'models':{n:{'run':str(p.relative_to(root)),'status':'completed','observations':len(records[n]),'expected':274,
                 'attempts':m['attempts'],'jobs':[{k:j[k] for k in ('name','kind','observations','exit_code')} for j in m['jobs']]}
           for n,p in runs.items() for m in [meta[n]]},
 'recovery':{'model':'mistral','prior_durable_rows':len(prior['prior_observations']),
 'prior_durable_rows_unchanged':True,'prepared_groups_at_interrupt':len(prior['prepared']),
 'cross_attempt_serial_pipeline_pairs':cross,'all_pair_bytes_and_first_tokens_identical':True,
 'interruption_exit_code':meta['mistral']['attempts'][0]['exit_code'],'resume_exit_code':0},
 'tests':{'shared_regression':86,'mistral_compatibility':3,'status':'passed'},
 'scope':'minimal functional matrix; no larger measurements executed'}
assert proof['recovery']['interruption_exit_code']==143
bundle=root/'reports/shared-v3-20261025'
bundle.mkdir(parents=True,exist_ok=True)

for n,run in runs.items():
 verify(run)
 extras={'v3-validation.json':json.dumps(proof,ensure_ascii=False,indent=2)+'\n',
         'SHARED_FRAMEWORK.md':Path('docs/reproduction/SHARED_FRAMEWORK.md').read_text(),
         'regression-tests.log':Path('.local/v3-regression-tests.log').read_text(),
         'mistral-compat-tests.log':Path('.local/v3-mistral-compat-tests.log').read_text(),
         'package_delivery.py':Path(__file__).read_text()}
 if n=='mistral':
  extras.update({'recovery-before.json':Path('.local/v3-recovery-before-03.json').read_text(),
                 'recovery-controller.py':Path('.local/v3-recovery-controller-03.py').read_text(),
                 'recovery-controller.log':Path('.local/v3-recovery-controller-03.log').read_text()})
 archived=json.loads((run/'artifacts.json').read_text())
 for name,value in extras.items():
  target=run/'validation'/name;target.parent.mkdir(exist_ok=True)
  if target.exists():
   assert target.read_text()==value,'Refusing to replace archived validation evidence'
  else: target.write_text(value)
  archived['files'][str(target.relative_to(run))]={'bytes':target.stat().st_size,'sha256':sha256(target)}
 save(run/'artifacts.json',archived)
 verify(run)
 for fig in (run/'figures').iterdir():
  target=bundle/'figures'/n/fig.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(fig,target)

save(bundle/'validation.json',proof)
reference={n:{'run':str(p.relative_to(root)),'artifacts_manifest_sha256':sha256(p/'artifacts.json'),
 'verified_files':verify(p),'summary_sha256':sha256(p/'summary.json')} for n,p in runs.items()}
save(bundle/'source-reference.json',reference)
(bundle/'code').mkdir(exist_ok=True);shutil.copy2(__file__,bundle/'code/package_delivery.py')
summary={n:json.loads((p/'summary.json').read_text()) for n,p in runs.items()}
def table(rows,cols):
 def fmt(x): return '—' if x is None else f'{x:.5g}' if isinstance(x,float) else str(x)
 return '\n'.join(['| '+' | '.join(cols)+' |','| '+' | '.join('---' for _ in cols)+' |']+['| '+' | '.join(fmt(r[c]) for c in cols)+' |' for r in rows])
text=['# 共享框架 v3 功能验收（2026-10-25）',
 '代码已接入分块串行／流水线、历史吞吐策略及轻量机制消融。TinyLlama 与 Mistral 使用逐文件相同的公共源码，在 RTX 4090 D 上各完成 **7/7 任务、274/274 正式观测**，退出码均为 0。以下为功能覆盖与一致性验证，质量与速度以逐项实测为准。',
 '两个最终批次均由 tmux 运行；Mistral 在 TTFT 阶段执行 TERM（143），同一代码与配置恢复后完成。此前提交的 '+str(len(prior['prior_observations']))+' 行 SHA 全部保持不变，'+str(cross)+' 组串行／流水线配对跨越了两次尝试，仍复用同样消息字节并生成相同首 token。详见 [机器验证记录](validation.json)。',
 '86 项公共回归与 3 项 Mistral 兼容检查通过；包括实际 CUDA 符号恢复、零值/小幅值、257-token 尾段、发送前预算后的接收可用时刻、队列阻塞、异常线程清理、历史信息隔离、位打包和稳定观测 ID。完整日志在两批原始目录的 validation/。',
 '## 范围与口径',
 '每模型质量54条；TTFT168条（4个文档/长度条件×2带宽×7方法×3模式）；四组倍率/SLO自适应共28条（7策略）；消融24条（1自然条件＋1事实的3位置，3精度×2熵路径）。另有196次网络任务预热。消融每行保存3次编解码计时及1次预热。自然文档、事实、问题、条件及重复分别计数；这是最小功能矩阵，不能据此推出大规模收益或统计显著性。',
 '应用块128 token，长度128/257覆盖单块及128＋128＋1尾块；原始CUDA内部最多256 token另记。原整包文本参考保留一次上下文＋问题forward；两个分块模式使用同一逐块前缀prefill和同一编码字节。发送、接收为CPU线程，GPU恢复/拼接/模型均在主线程；容量1的发送许可限制预取。',
 '历史吞吐首块固定1 Gbps，后续仅使用上一完整消息字节／有效接收区间。接收容量/上一块GPU等待不计入该区间，正常限速与socket开销计入。完整决策序列和各档预测成本在records.jsonl；当前策略仍为本地启发式。',
 '消融先共用一次原量化符号/尺度，实际4/5-bit打包与原GPU算术编码共用反量化。纯符号、CDF和完整容器分别计字节。统一16/32的名称指bins，实际4/5bit；分层2的位宽预算独立明示。量化阶段包含符号范围/尺度有效性验证，熵编码计时另存；这些少量设置展示体积—质量取舍。',
 '## 实测流水线配对（仅多块257-token条件）',
 '每行来自两个文档的配对速度比中位数，串行TTFT÷流水线TTFT；小于1表示流水线较慢。所有方法和两个带宽都逐项保留，不预设加速。']
for n,s in summary.items():
 text += [f'### {n}',table([r for r in s['pipeline_comparison'] if r['context_tokens']==257],['method','target_bandwidth_gbps','paired_observations','median_paired_speedup','mean_pipeline_overlap_seconds']),f'![{n}接收恢复时间线](figures/{n}/receive_restore_timeline.png)']
text += ['## 历史吞吐与当前带宽策略',
 '下表保留四组配置；单事实、单轨迹、单计时重复仅用于检查策略执行与记录。静态、三固定档及纯文本保留在完整CSV和原始报告。']
for n,s in summary.items():
 text += [f'### {n}',table([r for r in s['adaptive'] if r['strategy'] in ('adaptive','adaptive_history')],['task','strategy','median_ttft_seconds','slo_violation_rate','condition_mean_em'])]
text += ['## 机制消融',
 'packed/arithmetic两路径的符号、尺度、恢复KV、NLL和生成结果逐项相同。完整字节可能因CDF和容器开销高于直接打包；不能把纯符号收益当作完整空间收益。不同精度预算的质量变化不单独归因于分层。负ΔNLL本身不证明质量提升。']
for n,s in summary.items():
 text += [f'### {n}',table(s['ablation'],['precision','entropy','mean_symbol_bits','mean_symbol_data_bytes','mean_complete_bytes','paired_nll_delta','mean_em','mean_encode_excluding_quantize_seconds','mean_decode_seconds']),f'![{n}消融](figures/{n}/ablation.png)']
text += ['## 原始结果与操作',
 '\n'.join(f'- [{n}完整报告](../../results/{n}/minimal/v3-final-03/report.md) · [配置](../../results/{n}/minimal/v3-final-03/config.json) · [原始记录](../../results/{n}/minimal/v3-final-03/records.jsonl) · [时间线CSV](../../results/{n}/minimal/v3-final-03/ttft_timeline.csv)' for n in runs),
 '[运行、配置与tmux恢复指南](../../docs/reproduction/SHARED_FRAMEWORK.md) · [要求/代码/测试对应表](../../docs/reproduction/V3_CHANGE_AUDIT.md) · [归档来源及SHA](source-reference.json)。v1/v2及本次开发诊断不混入以上最终汇总；较大配置只有预检，没有启动测量。']
(bundle/'验收报告.md').write_text('\n\n'.join(text)+'\n')
files={str(p.relative_to(bundle)):{'bytes':p.stat().st_size,'sha256':sha256(p)} for p in sorted(bundle.rglob('*')) if p.is_file() and p.name!='report_manifest.json'}
save(bundle/'report_manifest.json',{'files':files})
print(json.dumps({'accepted':reference,'report_files':len(files),'recovery_cross_attempt_pairs':cross},indent=2))
