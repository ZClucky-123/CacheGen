"""Offline analysis of one immutable batch; never modifies original evidence.

Run from the repository root: python reports/mistral/20261026-larger-analysis/analyze.py
Only plotting needs matplotlib; all statistics use the standard library.
"""
import collections
import csv
import hashlib
import json
from pathlib import Path
import statistics as stats

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / 'results/mistral/larger/v3-larger-stable-20261025'
OUT = Path(__file__).resolve().parent


def write_csv(name, rows):
    with (OUT / name).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(str(v) for v in row) + ' |' for row in rows])


def main():
    manifest = json.loads((RUN / 'artifacts.json').read_text())
    for name, info in manifest['files'].items():
        data = (RUN / name).read_bytes()
        assert len(data) == info['bytes'], name
        assert hashlib.sha256(data).hexdigest() == info['sha256'], name
    rows = [json.loads(line) for line in (RUN / 'records.jsonl').open()]
    assert len(rows) == 12594
    assert len({r['record_id'] for r in rows}) == len(rows)
    summary = json.loads((RUN / 'summary.json').read_text())
    quality = summary['quality']
    ttft = summary['ttft']
    adaptive = summary['adaptive']
    # Check archived summary against observations, independent of the report text.
    for q in quality:
        selected = [r for r in rows if r['kind'] == 'quality' and
                    (r['context_tokens'], r['method']) == (q['context_tokens'], q['method'])]
        assert abs(stats.mean(r['nll'] for r in selected) - q['mean_nll']) < 1e-12
    for t in ttft:
        selected = [r for r in rows if r['kind'] == 'ttft' and
                    (r['context_tokens'], r['method'], r['mode'], r['bandwidth_gbps']) ==
                    (t['context_tokens'], t['method'], t['mode'], t['target_bandwidth_gbps'])]
        assert len(selected) == t['timing_observations'] == 50
        assert stats.median(r['ttft_wall_seconds'] for r in selected) == t['median_seconds']
    for a in adaptive:
        selected = [r for r in rows if r['kind'] == 'adaptive' and
                    (r['task'], r['strategy']) == (a['task'], a['strategy'])]
        assert len(selected) == a['timing_observations'] == 75
        assert stats.mean(r['slo_violation'] for r in selected) == a['slo_violation_rate']

    cold_start = []
    for a in adaptive:
        if a['strategy'] not in ('adaptive', 'adaptive_history'):
            continue
        selected = [r for r in rows if r['kind'] == 'adaptive' and
                    (r['task'], r['strategy']) == (a['task'], a['strategy'])]
        first = [r['decisions'][0] for r in selected]
        cold_start.append({
            'task': a['task'], 'strategy': a['strategy'], 'requests': len(selected),
            'first_level_counts': dict(collections.Counter(r['levels'][0] for r in selected)),
            'first_transfer_median_seconds': stats.median(d['transfer_seconds'] for d in first),
            'first_transfer_already_exceeds_slo': sum(d['transfer_seconds'] > r['slo_seconds']
                                                     for d, r in zip(first, selected)),
            'later_text_chunks': sum(level == 0 for r in selected for level in r['levels'][1:]),
            'initial_estimate_gbps': first[0]['estimate_gbps'] if a['strategy'] == 'adaptive_history' else None,
        })

    wire = []
    for length in (1024, 2048, 4096):
        for method in ('text', 'raw', 'uniform4', 'uniform8', 'cachegen1', 'cachegen2', 'cachegen3'):
            # Each source has identical bytes across repeats/bandwidths. Deduplicate.
            representations = {}
            for r in rows:
                if r['kind'] == 'ttft' and r['context_tokens'] == length and r['method'] == method:
                    key = (r['source_id'], r['mode'])
                    if key in representations:
                        assert representations[key] == r['bytes']
                    representations[key] = r['bytes']
            sizes = {mode: stats.mean(size for (source, m), size in representations.items() if m == mode)
                     for mode in ('whole', 'chunk_serial', 'chunk_pipeline')}
            assert sizes['chunk_serial'] == sizes['chunk_pipeline']
            wire.append({'context_tokens': length, 'method': method,
                         'whole_mean_mib': sizes['whole'] / 2**20,
                         'chunk_mean_mib': sizes['chunk_pipeline'] / 2**20,
                         'chunk_over_whole_bytes': sizes['chunk_pipeline'] / sizes['whole']})
    cross_baseline = []
    for t in ttft:
        if not t['method'].startswith('cachegen'):
            continue
        text = next(b for b in ttft if b['context_tokens'] == t['context_tokens'] and
                    b['target_bandwidth_gbps'] == t['target_bandwidth_gbps'] and
                    b['method'] == 'text' and b['mode'] == 'whole')
        cross_baseline.append({
            'context_tokens': t['context_tokens'], 'bandwidth_gbps': t['target_bandwidth_gbps'],
            'method': t['method'], 'mode': t['mode'], 'median_seconds': t['median_seconds'],
            'same_mode_text_speedup': t['speedup_vs_text'],
            'whole_text_speedup': text['median_seconds'] / t['median_seconds'],
            'whole_quality_gate_reference_only': t['same_length_quality_acceptance']})

    qa = [r for r in rows if r['kind'] == 'triviaqa']
    raw_qa = {r['case_id']: r for r in qa if r['method'] == 'raw'}
    qa_changes = []
    for r in qa:
        raw = raw_qa[r['case_id']]
        if r['em'] != raw['em']:
            qa_changes.append({k: r[k] for k in ('case_id', 'method', 'scored_answer', 'em')} |
                              {'raw_answer': raw['scored_answer'], 'raw_em': raw['em']})

    evidence = {'batch': str(RUN.relative_to(ROOT)), 'original_files_verified': len(manifest['files']),
                'formal_observations': len(rows), 'observation_kinds': dict(collections.Counter(r['kind'] for r in rows)),
                'original_manifest_sha256': hashlib.sha256((RUN / 'artifacts.json').read_bytes()).hexdigest(),
                'cold_start': cold_start, 'triviaqa_changes': qa_changes}
    (OUT / 'evidence.json').write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + '\n')
    write_csv('wire_sizes.csv', wire)
    write_csv('baseline_comparison.csv', cross_baseline)
    write_csv('triviaqa_changes.csv', qa_changes)

    def timing(length, bw, method, mode='whole'):
        return next(r for r in ttft if (r['context_tokens'], r['target_bandwidth_gbps'], r['method'], r['mode']) ==
                    (length, bw, method, mode))

    text = '''# Mistral 较大规模实验结果分析

本批次说明：KV 压缩显著减少了字节数；高速目标带宽下出现首 token 加速；相同分块的流水线有额外收益。但低带宽下文本重算更快，自适应策略也没有普遍优于固定档位。不能把这些结果概括为全部复现了论文的质量与加速结论。

分析仅使用 `v3-larger-stable-20261025`。运行配置为 RTX 4090 D、Mistral-7B FastChat 8bit 权重、FP16 KV，以及声明的稳定数值执行设置。正文数值由本目录 [analyze.py](analyze.py) 从本地原始数据生成；不需要 GPU，不改写原始归档。各输入/输出见文末。

## 1. 观测数量与统计单位

12594 条正式观测 = 360 条自然续写 + 540 条受控问答 + 120 条 TriviaQA + 9450 条 TTFT + 2100 条自适应 + 24 条消融。7/7 任务完成，退出码 0；175 项原始文件 SHA256 一致。程序重新核对全部 18 组质量均值、189 组 TTFT 中位数及 28 组自适应违规率。

这些不是 12594 个独立样本。自然续写为 20 个来源×3 长度；受控问答为 10 个事实×3 位置×3 长度；TriviaQA 为 20 题。TTFT 是 5 个来源×3 长度×3 带宽×7 方法×3 模式×10 次。自适应是 5 个事实×5 条轨迹×7 策略×2 倍率×2 SLO×3 次。既有区间按来源聚类抽样 2000 次，样本较少时用于描述，不构成质量等价证明。

## 2. 压缩：减少体积，但精度取舍不能省略

下面压缩比是完整序列化原始 FP16 KV 字节数÷完整压缩容器字节数；包括尺度、CDF 等元数据。ΔNLL 是同一来源相对 raw 的平均变化，越小越好。预设门槛是平均 ΔNLL≤0.02 且受控问答保留率≥90%。这是本框架门槛，并非论文统一验收标准。

'''
    qtable = []
    for r in quality:
        if r['method'].startswith('cachegen'):
            lo, hi = r['paired_nll_cluster_ci95']
            qtable.append([r['context_tokens'], r['method'], f"{r['raw_serialized_over_method']:.2f}×",
                           f"{r['paired_nll_delta']:.5f}", f'[{lo:.5f}, {hi:.5f}]',
                           '通过' if r['acceptance'] == 'pass' else '未通过'])
    text += table(['长度', '方法', '压缩比', '平均 ΔNLL', '描述性 95% 区间', '点估计门槛'], qtable)
    text += '''

CacheGen1 压缩最强，但 4096 token 的 ΔNLL=0.020898 超过门槛。该点的区间跨越 0.02，因此只能说本次点估计未通过，不能断言总体退化必然超过 0.02。CacheGen2 三种长度的点估计均通过，但 2048 token 的 ΔNLL=0.019415 已贴近门槛，且区间上限超过 0.02。CacheGen3 的三种长度区间上限均低于 0.02，在本批次自然续写上相对更稳健，体积也更大。负 ΔNLL 不应解释为压缩普遍提升模型能力。

uniform4 压缩约 3.98×、uniform8 约 2.00×，三种长度的现有点估计门槛均通过。不能只用 FP16 raw 对照证明 CacheGen 最优，还应在相近质量下比较 uniform4。

受控问答各方法都是 90/90 条件正确；它来自 10 个独立事实，说明该任务区分度有限。TriviaQA 的 raw 与 uniform8 是 16/20 正确，其余方法是 15/20。所有低分方法都少对 1 题，但丢失的题并不相同：

'''
    text += table(['题目 ID', '方法', 'raw 输出', '该方法输出'],
                  [[r['case_id'], r['method'], r['raw_answer'], r['scored_answer']] for r in qa_changes])
    text += '''

EM 按预先声明的答案别名评分，不能分析后手工扩大同义词来改变本次结果。CacheGen 的配对 EM 差均为 −5 个百分点，现有描述性区间为 [−15, 0] 个百分点：观察到少对一题，但小样本既不能证明总体无损，也不足以精确估计损失。

## 3. 传输：必须先选对文本基线

TTFT 包含 socket 建立、实际限速传输、接收端恢复/分词及首 token 在 CPU 可用之前的推理；不包括离线 prefill、编码和后续生成。所有结论都以已预先准备缓存为前提，不能直接用于只有一次访问且必须先建缓存的请求。

以整包 CacheGen2 对整包文本为例，以下为 50 次计时的 TTFT 中位数：

'''
    tt = []
    for length in (1024, 2048, 4096):
        for bw in (.1, 1, 10):
            c, t = timing(length, bw, 'cachegen2'), timing(length, bw, 'text')
            tt.append([length, bw, f"{t['median_seconds']*1000:.1f}",
                       f"{c['median_seconds']*1000:.1f}", f"{t['median_seconds']/c['median_seconds']:.2f}×"])
    text += table(['长度', '目标 Gbps', '整包文本 ms', '整包 CacheGen2 ms', '文本/CacheGen2'], tt)
    text += '''

低带宽时，压缩 KV 仍是数十 MiB，而文本只有数 KiB；即使省下 prefill，传输开销也可能更大。4096 token、0.1 Gbps 时，CacheGen2 平均传输约 7.223 秒；文本整段 TTFT 中位数约 0.822 秒。10 Gbps 目标下，CacheGen2 平均传输约 0.145 秒、恢复约 0.166 秒，才有机会优于约 0.817 秒的文本重算。这些相位是均值、TTFT 表是中位数，不能直接相加核对；流水线相位还存在重叠。

以 4096 token、10 Gbps 为例：整包文本 817.2ms、分块文本流水线 1397.3ms、CacheGen2 流水线 445.5ms。现有 `ttft.csv` 的 `speedup_vs_text=3.14×` 对应分块文本；改用较快的整包文本后约为 **1.83×**。两种数字各自有含义，但不能混用。新导出的 [baseline_comparison.csv](baseline_comparison.csv) 同时列出这两种口径。

同条件的 uniform4 流水线约 390.2ms，比 CacheGen2 流水线更快；但整包 uniform4 约 658.5ms，又慢于整包 CacheGen2 约 463.5ms。档位、分块和基线一起影响结论，不能从单个有利对照宣称方法普遍更优。

`same_length_quality_acceptance` 取自整包质量实验，只是同长度参考，并未重新评估 256-token 独立量化表示的完整 NLL/QA。分块重新计算尺度与 CDF，质量可能变化；全部 TTFT 首 token 与 raw 相同不等于完整回答质量相同。故分块速度可以测量和配对，但完整的质量—速度结论仍缺分块对应质量证据。

目标带宽由本地 socket 限速器设定，不等于真实物理链路容量；尤其 10 Gbps 下实际吞吐受消息、Python 与调度影响。现有数据只有 0.1/1/10 三个点，不能精确推出拐点带宽，也不能直接外推多机 WAN。

## 4. 流水线：有重叠，收益被分块体积抵消一部分

只有相同消息的 chunk_serial 与 chunk_pipeline 配对，才能单独衡量流水线收益。CacheGen2 的中位配对加速比如下（串行 TTFT÷流水线 TTFT）：

'''
    pipes = [r for r in summary['pipeline_comparison'] if r['method'] == 'cachegen2']
    text += table(['长度', '目标 Gbps', '配对加速比', '平均重叠 ms'],
                  [[r['context_tokens'], r['target_bandwidth_gbps'], f"{r['median_paired_speedup']:.3f}×",
                    f"{r['mean_pipeline_overlap_seconds']*1000:.1f}"] for r in pipes])
    text += '''

例如 4096 token、10 Gbps，CacheGen2 分块串行约 516.5ms、流水线约 445.5ms；配对中位加速约 1.135×。这证明受控实现中重叠确有收益，不代表相对整包方案同样能快 13.5%。

同样的 4096-token 输入，CacheGen2 整包平均传输 85.24MiB，256-token 分块合计 144.85MiB，多约 70%。分块各自携带尺度、CDF 和流长度元数据，也改变局部分布与量化；当前按包计算/发送 CDF，不能复用论文中的模型级共用分布。CDF 重复是可从实现确认的开销来源，但现有汇总没有对全部字节损失逐项归因，不把增量全部归给某一个字段。

由此可解释：分块串行较慢，流水线又追回部分时间；最终是否超过整包方案，取决于字节膨胀和重叠收益的平衡。详见 [wire_sizes.csv](wire_sizes.csv)。

## 5. 自适应：紧 SLO 下固定低档更好；历史策略有首块失配

以下每格是 SLO 违规率，每个策略/条件 75 次计时，来自 5 事实×5 轨迹×3 次重复。全部策略的本组受控 EM 都是 1。

'''
    tasks = ['adaptive-scale-1-slo-0.2', 'adaptive-scale-1-slo-0.5',
             'adaptive-scale-0.01-slo-0.2', 'adaptive-scale-0.01-slo-0.5']
    strategies = ['adaptive', 'adaptive_history', 'adaptive_static', 'fixed1', 'fixed2', 'fixed3', 'text']
    text += table(['策略', '倍率1 / 0.2s', '倍率1 / 0.5s', '倍率0.01 / 0.2s', '倍率0.01 / 0.5s'],
                  [[strategy] + [f"{next(r for r in adaptive if r['task']==task and r['strategy']==strategy)['slo_violation_rate']*100:.2f}%"
                                 for task in tasks] for strategy in strategies])
    text += '''

`adaptive` 读取当前块的轨迹带宽；`adaptive_history` 只用上一个完整消息的实测吞吐，首块固定猜测 1 Gbps。前者信息条件更有利，不能将其收益直接视为历史在线估计策略的收益。

倍率1、0.2秒时，fixed1 违规率只有 20%，而 adaptive 为 80%、history 约 81.3%。当前自适应优先尝试精度最高的可行档位，固定低档则牺牲精度预算换速度；本次受控 EM 没有区分这些精度差别。结果说明本次策略没有在紧 SLO 下取得最好的实测达标率，不代表固定低档在所有自然任务都满足相同质量要求。也不能只和 100% 违规的文本对照来夸大自适应优势。

低带宽倍率0.01时，当前带宽策略的 75 次请求均选择四块文本；历史策略则出现一致的首块问题：

'''
    cold = [r for r in cold_start if 'scale-0.01-' in r['task'] and r['strategy'] == 'adaptive_history']
    text += table(['SLO', '首块选择', '首块传输中位数', '仅首块已超时', '后三块回退文本'],
                  [[r['task'].rsplit('-', 1)[1]+'s', str(r['first_level_counts']),
                    f"{r['first_transfer_median_seconds']:.3f}s", f"{r['first_transfer_already_exceeds_slo']}/75",
                    f"{r['later_text_chunks']}/225"] for r in cold])
    text += '''

这是逐块日志直接支持的解释：初始 1 Gbps 估计下选择 KV，真实首块传输已消耗 1.03/1.79 秒，随后即使三块全退回文本也无法追回 0.2/0.5 秒的期限。宽松 SLO 在这个实现和低带宽设置中反而允许选择更大的首块，最终更慢。它不是压缩器不能恢复 KV，而是初始带宽假设、选档和不可撤销首块传输的组合问题。现有数据没有运行新的初值/探测/超时中止对照，不能声称某个修复已经验证有效。

倍率0.01、0.2秒时，文本也 100% 超时（中位数约 0.384秒）；因此该条件下这些已测策略都没满足目标。0.5秒时文本本来已全部达标，自适应回退文本没有额外的达标率收益。

## 6. 消融：熵编码的体积贡献有直接证据，分层优势仍未被单独识别

12 组打包/算术编码配对共用完全相同的量化符号和尺度，恢复与质量结果一致。完整容器的额外压缩为 1.29–1.73×，说明熵编码在这些小样本上确实节省字节。仅比较符号流会高估收益，因为 CDF 等元数据也需要发送。

但本组只有 1 个自然来源、1 个事实的 3 个位置。uniform16/32 实际为 4/5-bit 符号预算，layered2 平均约 4.1875 bit；不能把不同位预算的 NLL 差异全部归因于分层分配。统一位打包路径主要在 CPU/NumPy 执行，算术编码路径使用 CUDA，计时差同时含实现和设备因素，不是纯算法加速比较。

## 7. 这批结果支持什么，后续先做什么

| 结论 | 本批次证据 |
| --- | --- |
| 完整 KV 容器可明显缩小 | 支持；三档约 4.13–7.48×，需附档位与质量条件 |
| 高速目标带宽下缓存复用可减少 TTFT | 支持部分条件；例如整包 CacheGen2 在 10 Gbps 下约 1.56–1.76× |
| 同字节流水线有额外收益 | 支持；与相同分块串行配对比较 |
| 压缩始终优于文本重算 | 不支持；低带宽明显更慢 |
| 自适应优于所有固定策略且总能满足 SLO | 不支持；紧 SLO 的 fixed1 更好，低带宽 history 首块就超时 |
| 质量完全不变，且所有档位通过门槛 | 不支持；4096/CacheGen1 未过点估计门槛，TriviaQA 丢失题目 |
| 256-token 分块质量等同整包质量 | 证据不足；已有首 token/同字节正确性不能替代完整分块 NLL/QA |
| 已复现论文全部倍数及完整方法 | 不支持；模型/硬件/文本基线不同，anchor/delta、模型级共用 CDF 和完整论文调度器未实现 |

建议先完成现有数据的论文式展示：分别呈现整包端到端收益、同分块流水线收益和 SLO—质量取舍；不合并不同基线，不调整既定门槛。若以后需要新增 GPU 实验，优先补分块 NLL/真实 QA、历史策略冷启动对照，以及匹配总位预算的分层消融，再决定是否扩展规模。服务器关闭不影响当前全部离线核验、统计与绘图。

## 文件与离线复算

原始批次：[report.md](../../../results/mistral/larger/v3-larger-stable-20261025/report.md)、[quality.csv](../../../results/mistral/larger/v3-larger-stable-20261025/quality.csv)、[ttft.csv](../../../results/mistral/larger/v3-larger-stable-20261025/ttft.csv)、[adaptive.csv](../../../results/mistral/larger/v3-larger-stable-20261025/adaptive.csv)。原始大文件通过该批次 README 的分卷命令恢复。边界见 [论文对照](../../../docs/reproduction/PAPER_ALIGNMENT.md)。

新增输出：[核验与首块证据](evidence.json)、[字节数](wire_sizes.csv)、[双基线对照](baseline_comparison.csv)、[TriviaQA 差异](triviaqa_changes.csv)。所有图表是描述性图，不把样本量或计时重复放大为独立证据。

```bash
python3 reports/mistral/20261026-larger-analysis/analyze.py
```

复算需 Python 标准库与 matplotlib；读取现成报告无需安装环境。脚本会先核验 175 项原始文件，缺少大文件时应先恢复归档。图表标注使用英文便于跨平台字体显示，PNG 与四页 PDF 均可独立分享。

![压缩和自然续写质量](quality.png)

![相对整段文本的 TTFT 收益](ttft.png)

![流水线收益和分块体积](pipeline.png)

![自适应违规率](adaptive.png)

[四页图表 PDF](analysis-figures.pdf)
'''
    (OUT / '实验结果分析.md').write_text(text, encoding='utf-8')
    plot(summary, wire, cross_baseline, tasks, strategies)
    print(json.dumps({'verified_original_files': len(manifest['files']), 'observations': len(rows),
                      'report': str((OUT / '实验结果分析.md').relative_to(ROOT))}, ensure_ascii=False))


def plot(summary, wire, comparisons, tasks, strategies):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import numpy as np
    colors = {'cachegen1': '#ce6a32', 'cachegen2': '#287b8e', 'cachegen3': '#795b91'}
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    with PdfPages(OUT / 'analysis-figures.pdf') as pdf:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), layout='constrained')
        for ax, length in zip(axes, (1024, 2048, 4096)):
            for method, color in colors.items():
                r = next(q for q in summary['quality'] if q['context_tokens'] == length and q['method'] == method)
                y = r['paired_nll_delta']; lo, hi = r['paired_nll_cluster_ci95']
                ax.errorbar(r['raw_serialized_over_method'], y, yerr=[[y-lo], [hi-y]],
                            fmt='o', capsize=4, color=color, label=method)
            ax.axhline(.02, ls='--', color='#a83a3a', label='Mean gate = 0.02')
            ax.axhline(0, color='gray', lw=.6)
            ax.set(title=f'{length} tokens; 20 source documents', xlabel='Full-container compression ratio',
                   ylabel='Paired NLL change (lower is better)', ylim=(-.018, .045))
            ax.grid(alpha=.18)
        axes[0].legend(fontsize=8)
        savefig(fig, 'quality', pdf, plt)

        fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), layout='constrained')
        for ax, length in zip(axes, (1024, 2048, 4096)):
            for method, color in colors.items():
                for mode, style in [('whole', '-'), ('chunk_pipeline', '--')]:
                    rs = sorted((r for r in comparisons if r['context_tokens'] == length and
                                r['method'] == method and r['mode'] == mode), key=lambda r:r['bandwidth_gbps'])
                    ax.plot([r['bandwidth_gbps'] for r in rs], [r['whole_text_speedup'] for r in rs],
                            style, marker='o', color=color, label=method+'/'+mode)
            ax.axhline(1, color='gray', lw=1)
            ax.set(xscale='log', yscale='log', title=f'{length} tokens; 5 sources x 10 repeats',
                   xlabel='Socket target bandwidth (Gbps)', ylabel='Whole-text median / method median', xticks=[.1,1,10])
            ax.set_xticklabels(['0.1','1','10']); ax.grid(alpha=.18)
        axes[0].legend(fontsize=6)
        fig.suptitle('Descriptive speed ratios; quality is separate (chunk quality not fully evaluated)', fontsize=10)
        savefig(fig, 'ttft', pdf, plt)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout='constrained')
        for bw in (.1, 1, 10):
            rs = [r for r in summary['pipeline_comparison'] if r['method']=='cachegen2' and r['target_bandwidth_gbps']==bw]
            axes[0].plot([r['context_tokens'] for r in rs], [r['median_paired_speedup'] for r in rs], marker='o', label=f'{bw:g} Gbps')
        axes[0].axhline(1, color='gray', lw=1)
        axes[0].set(title='CacheGen2: same-payload paired pipeline benefit', xlabel='Context tokens', ylabel='Median(serial TTFT / pipeline TTFT)')
        axes[0].legend(); axes[0].grid(alpha=.18)
        rs = [r for r in wire if r['method']=='cachegen2']; x = np.arange(len(rs))
        axes[1].bar(x-.18, [r['whole_mean_mib'] for r in rs], width=.36, label='Whole')
        axes[1].bar(x+.18, [r['chunk_mean_mib'] for r in rs], width=.36, label='256-token chunks')
        axes[1].set(title='CacheGen2: complete transmitted containers', xticks=x,
                    xticklabels=[str(r['context_tokens']) for r in rs], xlabel='Context tokens', ylabel='Mean bytes (MiB)')
        axes[1].legend(); savefig(fig, 'pipeline', pdf, plt)

        fig, ax = plt.subplots(figsize=(10, 5), layout='constrained')
        values = np.array([[next(r['slo_violation_rate'] for r in summary['adaptive'] if r['strategy']==strategy and r['task']==task)*100
                            for task in tasks] for strategy in strategies])
        im=ax.imshow(values, vmin=0, vmax=100, cmap='YlOrRd', aspect='auto')
        for i in range(len(strategies)):
            for j in range(len(tasks)):
                ax.text(j, i, f'{values[i,j]:.1f}%', ha='center', va='center', color='white' if values[i,j]>65 else 'black')
        ax.set(xticks=range(4), xticklabels=['Scale 1\nSLO 0.2s','Scale 1\nSLO 0.5s','Scale 0.01\nSLO 0.2s','Scale 0.01\nSLO 0.5s'],
               yticks=range(7), yticklabels=strategies, title='SLO violation rate; lower is better\n75 timings per cell; 5 independent facts x 5 traces x 3 repeats')
        fig.colorbar(im, ax=ax, label='Violation rate (%)');savefig(fig, 'adaptive', pdf, plt)


def savefig(fig, name, pdf, plt):
    fig.savefig(OUT / (name+'.png'), dpi=180)
    pdf.savefig(fig)
    plt.close(fig)


if __name__ == '__main__':
    main()
