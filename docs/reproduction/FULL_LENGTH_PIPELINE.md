# 全长真实传输：分块与流水线对照

2026-11-25已核对：正式1680/1680条完成、退出码0、用时约71分钟；65个下载文件哈希匹配。结果见 [归档报告](../../reports/mistral/20261124-fulllength-pipeline/README.md)。以下为预先固定协议与历史启动记录。

2026-11-24预先固定协议；只在服务器执行，不在本机运行模型或测试。不修改既有实验数据。

目的：补齐全长上下文的实际传输与解码重叠证据，不把“体积/带宽”估算或理想流水下限当作实测。

## 固定设置

- Mistral-7B FP16、4090D；沿用当前稳定计算实现，LongChat ID0–19，完整上下文不截断，问题由接收端现算。
- 作者公开codec低精度cachegen1与高精度cachegen3，固定精度，不使用自适应；此路径没有差分。只回答作者公开实现的传输表现。
- 带宽0.5/1/3/10Gbps，真实限速socketpair；不是WAN。每个条件计时3次，每文档先取中位数，再跨文档汇总；加速比先逐文档配对再取中位数。
- 三种模式：整份编码后串行传输与恢复、1536 token分块串行、相同分块包的传输—解码流水线。接收队列容量1，限速量子65536字节。
- 额外提供整段文本重算基线（仅whole模式）。20×(2×3+1)×4×3 = 1680条计时记录，独立文档20篇。ID0试跑84条，试跑不并入正式结果。
- 每种表示/模式先做一次无限速socket预热，不计入正式记录。顺序按固定随机种子逐文档打散，保存实际顺序。

## 公平性与边界

分块串行和流水线使用完全相同的字节包，预热时检查恢复后KV的SHA256和首token一致，计时重复检查字节及首token。仅这一对照能单独归因于重叠执行。

整份与分块重新编码的概率表、长度字段和元数据开销可能不同，因此二者比较是分块编码加调度的净效果，不是只改变传输调度的纯消融。额外计入真实包大小，分别保存各表示的准备元数据。

TTFT从socket建立前、CUDA同步后开始，到原始问题现算并获得CPU可见首token为止。没有带宽探测；本轮不做策略决策。模型加载、上下文prefill、编码、预热、质量评分均在计时外。与有探测的自适应实验不能混成同一计时协议。

传输与恢复的起止时间逐块记录，保存实际重叠区间。网络活跃时长、恢复时长会重叠，不能直接相加当作TTFT；报告总TTFT、问题首token耗时、实际重叠和其他开销。

质量在预热后独立生成最多20 token，记录话题准确率、逐字一致率、相对原始回答token的条件NLL和KV指纹；不混用上下文续写NLL。计时重复不再次生成完整回答，仅验证首token一致。本文本重算采用当前稳定计算实现，不能称为vLLM最优基线。

## 产物和运行

配置：`paper_compression/configs/full-length-pipeline.json`。入口：`tools/full_length_pipeline.py`。所有输出目录必须不存在，失败写状态并停止，不覆盖、不自动混合旧结果。

服务器先运行新wrapper测试及已有`StreamingTests`，再用ID0试跑。试跑应满足84条完整计时矩阵、字节一致性、串行/流水线恢复指纹一致、串行无重叠、流水线确有重叠，才能启动全量1680条。若失败，保留证据，先修复验证。

后台任务使用nohup、日志、PID、exitcode，持有已有GPU互斥锁；原始数据和源码/模型/数据指纹保存在对应新目录。所需时间以试跑实测估算，不提前承诺收益或总时长。

```bash
source scripts/experiment-env.sh
cachegen_lock_gpu
"$CACHEGEN_PYTHON" -m unittest discover -s tests -p test_full_length_pipeline.py -v
PYTHONPATH="tests:$PYTHONPATH" "$CACHEGEN_PYTHON" -m unittest test_v3_mechanisms.StreamingTests -v
"$CACHEGEN_PYTHON" -u -m tools.full_length_pipeline --ids 0 --output .local/full-length-pipeline-20261124/pilot
"$CACHEGEN_PYTHON" -u -m tools.full_length_pipeline --output .local/full-length-pipeline-20261124/full
```

## 本次启动记录

2026-11-24确认服务器4090D空闲后同步代码；新增wrapper两项测试及现有传输四项测试均在服务器通过。本机未运行测试、模型或CUDA编译。

后台控制器：`.local/full-length-pipeline-20261124/launch.sh`，启动PID21628，持有GPU锁。`pilot.log`、`pilot.pid`、`pilot.exitcode`记录试跑；仅当84条试跑完整、字节检查与KV指纹一致性通过、流水线实际存在重叠后才自动启动全量。正式结果为`full/`，另有`full.log`、`full.pid`、`full.exitcode`。控制器日志为`launcher.log`。来源快照保存在该目录的`source-snapshot.tar.gz`，运行元数据同时记录源码、数据与模型指纹。

启动时仅确认试跑正在执行，不提前判定收益。启动完整批次后的状态和时间估计以服务器日志为准。

后续检查：ID0试跑84/84完成，退出码0，用时219.96秒；全部启动检查通过，控制器输出`PILOT_CHECKS_PASSED`。全量1680条已启动，状态running。按单篇试跑估算完整批次约1–1.5小时。试跑只用于正确性与运行时间估算，不据单篇数据宣称性能结论；启动检查不要求流水线必须更快。
