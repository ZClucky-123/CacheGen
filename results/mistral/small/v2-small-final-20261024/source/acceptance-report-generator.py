"""Write a Chinese acceptance note only after all archived observations pass."""
import json
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from experiments.artifacts import verify, sha256


def table(rows, columns):
    def fmt(v):
        if v is None:
            return '不可计算'
        return f'{v:.5g}' if isinstance(v, float) else str(v)
    return '\n'.join(['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join('---' for _ in columns) + ' |'] +
                     ['| ' + ' | '.join(fmt(row[c]) for c in columns) + ' |' for row in rows])


def main():
    run = REPO / 'results/mistral/small/v2-small-final-20261024'
    tiny = REPO / 'results/tinyllama/minimal/v2-final-check'
    manifest = json.loads((run / 'manifest.json').read_text())
    tm = json.loads((tiny / 'manifest.json').read_text())
    assert manifest['status'] == tm['status'] == 'completed'
    assert manifest['expected_observations'] == 552 and tm['expected_observations'] == 134
    assert len(manifest['jobs']) == len(tm['jobs']) == 6
    assert all(j['status'] == 'completed' for j in manifest['jobs'] + tm['jobs'])
    verified = {'mistral': verify(run), 'tinyllama': verify(tiny)}
    summary = json.loads((run / 'summary.json').read_text())
    cfg = json.loads((run / 'config.json').read_text())
    selection = json.loads((run / 'selection.json').read_text())
    a = json.loads((tiny / 'metadata.json').read_text())['identity']['inputs']['code']
    b = json.loads((run / 'metadata.json').read_text())['identity']['inputs']['code']
    assert a == b, 'The final model validations must use exactly the same repository code'
    proof = {'tiny_run': str(tiny.relative_to(REPO)), 'mistral_run': str(run.relative_to(REPO)),
             'all_repository_sources_match': True, 'code_files': len(a), 'sha256': a,
             'artifact_files_verified': verified}
    (REPO / 'docs/reproduction/shared-helper-validation-20261024.json').write_text(json.dumps(proof, indent=2) + '\n')
    history = json.loads((REPO / '.local/history-integrity.json').read_text())['files_sha256']
    assert all(sha256(REPO / p) == expected for p, expected in history.items())
    q = [r for r in summary['quality'] if r['method'] == 'cachegen3']
    timing = [r for r in summary['ttft'] if r['method'] == 'cachegen3']
    tqa = {r['method']: r for r in summary['qa'] if r['dataset'] == 'triviaqa'}
    calibration = json.loads((run / 'calibration.json').read_text())['rows']
    high = [r['actual_gbps'] for r in calibration if r['payload_bytes'] == 8388608 and r['target_gbps'] == 10]
    checks = json.loads((run / 'cache_checks.json').read_text())
    tz = timezone(timedelta(hours=8))
    started = datetime.fromisoformat(manifest['started_utc']).astimezone(tz).isoformat()
    finished = datetime.fromisoformat(manifest['finished_utc']).astimezone(tz).isoformat()
    jobs = [{'任务': j['name'], '状态': j['status'], '正式观测': j['observations']} for j in manifest['jobs']]
    quality = [{'长度': r['context_tokens'], '方法': r['method'], 'FP16序列化/方法': r['raw_serialized_over_method'],
                '配对ΔNLL': r['paired_nll_delta'], 'raw正确题': r['raw_em_correct'],
                '正确题保留率': r['em_retention'], '门槛': r['acceptance']} for r in summary['quality']]
    text = [
        '# 公共框架与完整小规模验收（2026-10-24）',
        f'最终 Mistral 批次：`{run.relative_to(REPO)}`；北京时间 {started} 至 {finished}。6/6 核心任务、552/552 正式观测完成，退出码 0，{verified["mistral"]} 项归档文件通过 SHA-256 校验。',
        f'TinyLlama 在相同源码下完成最小检查 6/6 任务、134/134 观测，{verified["tinyllama"]} 项归档文件通过校验。它是功能检查批次，不充作 TinyLlama 较大规模结果。',
        '## 是否与 TinyLlama 对齐',
        '框架与覆盖已对齐：六种体积质量方法；自然续写；开头/中间/末尾×多长度受控 QA；独立真实 TriviaQA；TTFT；原六策略与四组倍率×SLO。NLL 与 QA 规则共同升级到 cachegen-shared-v2。两个模型实际执行的全部仓库辅助源码指纹完全相同，见 [源码核对](shared-helper-validation-20261024.json)。',
        '三组实验仍是六个执行任务，未改造成六类实验。',
        table(jobs, ['任务', '状态', '正式观测']),
        '## 样本与重复的区别',
        '质量：2 个独立 LongChat 源文档×2 长度＝4 个续写条件；2 个事实×3 位置×2 长度＝12 个受控条件；4 个完整 TQA 问题。每条件比较六方法，共 120 条逐方法观测。',
        'TTFT：同 2 源文档×2 长度×3 带宽×4 方法×3 正式重复＝144 次；48 次预热另存且排除。自适应：3 个事实×2 轨迹×6 策略×2 重复＝72 次/配置，共四配置 288 次；另有 144 次预热。轨迹、长度、位置、方法和计时重复均不是新的独立问题；自适应事实包含质量组的前两个事实。',
        '## 质量与实际序列化体积',
        table(quality, ['长度', '方法', 'FP16序列化/方法', '配对ΔNLL', 'raw正确题', '正确题保留率', '门槛']),
        '门槛固定为平均配对 ΔNLL≤0.02 且保留≥90% 同长度 raw EM 正确题。零 raw 正确题时为 null/unassessable，相关行为有回归测试。标准 EM/F1 与短语诊断分开，没有临时追加别名或放宽门槛。',
        f'独立 TQA（source IDs：{[r["source_id"] for r in selection["triviaqa"]]}）：raw EM/F1={tqa["raw"]["mean_em"]:.3f}/{tqa["raw"]["mean_f1"]:.3f}，CacheGen 3={tqa["cachegen3"]["mean_em"]:.3f}/{tqa["cachegen3"]["mean_f1"]:.3f}。受控事实结果不能替代这组真实 QA，不能据小样本声称普遍质量等价。',
        'NLL 评分全部 held-out token 1..16，记录目标 token IDs 与逐 token 损失；最后上下文 token 对恢复的 KV[0:L-1] 重放。负 ΔNLL 不直接解释为模型改善。配对区间按独立源文档/事实/问题聚类；共享长度保持相关，重复计时不当作独立样本，小样本 P95 仅描述。',
        '## TTFT、网络与缓存正确性',
        f'CacheGen 3 在 {len(timing)} 个长度/带宽组合中，{sum(r["speedup_vs_text"]>1 for r in timing)} 个组合的中位 TTFT 优于优化后的文本重算，{sum(r["speedup_vs_uniform8"]>1 for r in timing)} 个组合优于 Uniform8。速度比较仍须同时检查对应长度的质量门槛；不能由这些数值宣称论文级收益。',
        f'10Gbps 目标、8MiB payload 校准实测 {min(high):.3f}–{max(high):.3f}Gbps；小包开销更明显。实际吞吐逐次保存，没有将目标 10Gbps 写成实测 10Gbps，也没有限速 SSH 主网卡。',
        f'缓存检查在 {len(checks["cases"])} 个真实文档/长度条件上比较完整 forward、只投影末位置的实现、raw KV 复用、重复生成、自然续写、分块拼接及上下文相关文本重算。raw logits 最大误差 {max(r["raw_logits_max_error"] for r in checks["cases"]):.6g}；多 token greedy 输出一致，原缓存未被污染。另对所有压缩方法检查形状/FP16/有限值与质量，GPU 测试逐符号覆盖 257-token 尾块和解码缓冲区扩容。',
        '全部正式观测真实执行 GPU 推理与实际字节 socket 传输；网络为本机限速模拟，不是跨服务器 WAN。TTFT 为完整接收后恢复，adaptive 为逐块串行；没有传输—解码流水线。trace_estimate 模式仅实现，没有混入本批次。',
        '文本采用 UTF-8＋显式 token 修补元数据，接收端实际重分词，framing 与 patch 全计入字节。离线 prefill、编码及自适应准备分别保存，作为每次上下文准备的注记，不能按计时重复数重复累加。',
        '## 自适应结果',
        table(summary['adaptive'], ['task', 'strategy', 'mode', 'timing_observations', 'median_ttft_seconds', 'slo_violation_rate', 'condition_mean_em', 'raw_em_retention', 'text_fallback_chunks']),
        '策略仍是 adaptive、adaptive_static、fixed1/2/3、text。仅用当前带宽预测剩余成本；text 回退实传、重分词并在已拼接前缀上真实计算。缓存块由完整原上下文计算后切分，保留原块间上下文依赖；尾块与 cache_position 显式检查。SLO 对应首 token，完整回答延迟另存。',
        '## 代码、历史和归档',
        '职责与合并细节见 [审查表](SHARED_FRAMEWORK_AUDIT.md)；新公共代码在 experiments/，数据/量化/传输/策略复用 mini_repro/。Mistral 原专用实验器的重复模型加载、编解码、评分、QA/NLL、TTFT 与汇总由公共实现替代；薄兼容入口保留。旧 TinyLlama v1 API/协议与历史报告器为核查保留。没有删除历史记录、权重或用户数据。',
        f'{len(history)} 个此前已归档的 TinyLlama/Mistral 原始文件与报告保持原始 SHA-256。历史路径与限制见 [历史索引](HISTORY.md)，未移动或重评分旧主指标。',
        '本轮缺权重预检失败、报告绘图依赖修复和受控中断恢复记录均保留。`v2-small-20261024` 因漏记离线 prefill 被替代，原记录与 superseded.json 保留；主结论仅来自新的 v2-small-final-20261024，没有拼接两种代码版本的观测。',
        '完整归档包含 config/protocol、模型/revision/权重/tokenizer SHA、Git HEAD/补丁和全部辅助源码、依赖/CUDA/驱动/GPU/CPU/内存限制、数据/选择、SQLite 检查点、逐例/顺序/生成、日志、CSV/JSON、图表和文件校验。',
        '## 直接运行较大规模',
        '两份 larger 配置只执行了数据/窗口/资源/绘图预检，未自动启动正式测量，见 [Mistral 检查](larger-mistral-preflight-20261024.json)与 [TinyLlama 检查](larger-tinyllama-preflight-20261024.json)。Mistral 4096-token 矩阵采用原 FastChat 8bit 权重，预估约 15.13GiB GPU、20.21GiB 主存；实际预检会重新核对可用资源。建议至少 2GiB 额外磁盘，预计数小时，取决于带宽与矩阵。TinyLlama 受 2048 窗口和仅 7 个短 TQA 的限制。这不是单张 4090 的官方 70B 配置。',
        '```bash\npython3 -m experiments.background start --name mistral-larger-new -- \\\n  --config experiments/configs/larger-mistral.json --run-id larger-new\npython3 -m experiments.background status --name mistral-larger-new\npython3 -m experiments status mistral --run-dir results/mistral/larger/larger-new\npython3 -m experiments.background start --name mistral-larger-resume -- \\\n  --resume results/mistral/larger/larger-new\n```',
        'tmux 在服务器持有进程；关闭终端可继续。`tmux -L cachegen attach -t <会话名>` 查看，Ctrl-b d 脱离。恢复前核对配置、源码、数据、模型与环境；同 GPU 使用共用排他锁，版本变化必须新批次。完整操作见 [运行指南](SHARED_FRAMEWORK.md)。',
        '## 原始证据与图表',
        '[完整通用报告](../../results/mistral/small/v2-small-final-20261024/report.md) · [逐例结果](../../results/mistral/small/v2-small-final-20261024/records.jsonl) · [配置](../../results/mistral/small/v2-small-final-20261024/config.json) · [计时顺序与预热](../../results/mistral/small/v2-small-final-20261024/events.jsonl) · [源码](../../results/mistral/small/v2-small-final-20261024/source/) · [校验清单](../../results/mistral/small/v2-small-final-20261024/artifacts.json)',
        '![体积与质量](../../results/mistral/small/v2-small-final-20261024/figures/quality.png)',
        '![TTFT](../../results/mistral/small/v2-small-final-20261024/figures/ttft.png)',
        '![自适应](../../results/mistral/small/v2-small-final-20261024/figures/adaptive.png)',
        '每图同时提供独立 SVG/PDF。完成覆盖仅证明本仓库代理实验执行完整；当前编码器缺少论文 anchor/delta 路径，本轮无远程网络或流水线，不构成完整论文复现。',
    ]
    path = REPO / 'docs/reproduction/SHARED_FRAMEWORK_RESULTS_20261024.md'
    path.write_text('\n\n'.join(text) + '\n', encoding='utf-8')
    print(path)


if __name__ == '__main__':
    main()
